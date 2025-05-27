#!/usr/bin/env python3
# simple_explore.py
# =========================================================
# Minimal repro of Simple Explore agents with fixed
# queue / worker logic (no more dead‑locks on batch_size>1)
# =========================================================
import json
import asyncio
import uuid
from collections import deque
from pathlib import Path
from typing import Any, List, Dict, Optional, Set, Callable, Tuple
import os
import pandas as pd
import tempfile
import time

import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import torch.nn.functional as F
from transformers import AutoTokenizer

import openhands
import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.core.logger import openhands_logger as logger
from openhands.utils.async_utils import call_sync_from_async, call_async_from_sync
from openhands.llm.llm import LLM
from openhands.core.config import LLMConfig, AgentConfig
from openhands.core.message import Message, TextContent
import re

# ---------------------------------------------------------
# TEMPLATES + HELPER FUNCTIONS
# ---------------------------------------------------------
SIMPLE_EXPLORE_TEMPLATE = """
    <prompt>
    {prompt}
    </prompt>

    <history>
    {history}
    </history>
    
    Respond in the following format, using careful step-by-step reasoning.
    The <history> tag contains the history of previous attempted answers, which you can use to inform your reasoning.
    It is critical that the answer provided in the <answer> tag is DIFFERENT from the previous attempts; all that matters is that one of the total attempts is correct.

    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

chat_template = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% generation %}"
    "{{message['content'] + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\\n'}}"
    "{% endif %}"
    "{% endfor %}"
)

chat_template_qwen3_thinking = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\\n'}}"
    "{% generation %}"
    "{% set full_content = message['content'] %}"
    "{% set mycontent = message['content'] %}"
    "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
    "{% if '</think>' in full_content and not is_last_message %}"
    "{% set mycontent = full_content.split('</think>')[-1].lstrip('\\n') %}"
    "{% endif %}"
    "{{mycontent + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\\n'}}"
    "{% endif %}"
    "{% endfor %}"
)

# ---------------------------------------------------------
# Padding helpers (unchanged)
# ---------------------------------------------------------
def convert_right_padding_to_left(tokenizer, input_ids, attention_mask, device, max_len=None):
    batch_size, orig_seq_length = input_ids.size()
    seq_length = max_len if max_len is not None else orig_seq_length
    left_padded_input_ids = torch.full(
        (batch_size, seq_length), tokenizer.pad_token_id,
        dtype=input_ids.dtype, device=device
    )
    left_padded_attention_mask = torch.zeros(
        (batch_size, seq_length), dtype=attention_mask.dtype, device=device
    )
    for i in range(batch_size):
        seq_len = attention_mask[i].sum().item()
        seq_len = min(seq_len, seq_length)
        offset = seq_length - seq_len
        left_padded_input_ids[i, offset:] = input_ids[i, :seq_len]
        left_padded_attention_mask[i, offset:] = 1
    return left_padded_input_ids, left_padded_attention_mask


def pad_to_max_length_right(tokenizer, encodings, max_length, device):
    batch_size = len(encodings["input_ids"])
    padded_input_ids = torch.full(
        (batch_size, max_length), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    padded_attention_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )
    padded_assistant_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )
    num_trimmed = 0
    for i in range(batch_size):
        seq_len = encodings["attention_mask"][i].sum().item()
        actual_len = min(seq_len, max_length)
        if seq_len > max_length:
            logger.warning(
                f"Trimming sequence length from {seq_len} to {actual_len} for batch item {i}"
            )
            num_trimmed += 1
        padded_input_ids[i, :actual_len] = torch.tensor(
            encodings["input_ids"][i][:actual_len], device=device
        )
        padded_attention_mask[i, :actual_len] = torch.tensor(
            encodings["attention_mask"][i][:actual_len], device=device
        )
        padded_assistant_mask[i, :actual_len] = torch.tensor(
            encodings["assistant_masks"][i][:actual_len], device=device
        )
    logger.info(
        f"Trimmed {num_trimmed*100/max(batch_size,1):.1f}% of samples in the batch of size {batch_size}"
    )
    return padded_input_ids, padded_attention_mask, padded_assistant_mask


# =========================================================
# AGENT CLASSES
# =========================================================
class SimpleExploreAgent:
    def __init__(
        self,
        instance_id: int,
        trajectory_id: int,
        prompt: str,
        max_prompt_length: int = 1024,
        infer_engine=None,
        tokenizer=None,
        sampling_params=None,
        max_iterations: int = 10,
        reward_func: Callable = None,
        qwen3_enable_thinking: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.step_count = 0
        self.infer_engine = infer_engine
        self.sampling_params = sampling_params
        self.instance_id = instance_id
        self.trajectory_id = trajectory_id
        self.max_iterations = max_iterations
        self.prompt = prompt
        self.reward_func = reward_func
        self.qwen3_enable_thinking = qwen3_enable_thinking
        self.history: List[str] = []

    def _prepare_input(self) -> str:
        formatted_history = "\n".join(
            [f"Turn {i}: {answer}" for i, answer in enumerate(self.history)]
        )
        return SIMPLE_EXPLORE_TEMPLATE.format(
            prompt=self.prompt, history=formatted_history
        )

    def _parse_response(self, response_str: str) -> Optional[str]:
        m = re.search(r"<answer>(.*?)</answer>", response_str, flags=re.DOTALL)
        return m.group(1).strip() if m else None

    async def generate(self, prompt, sampling_params):
        res = await self.infer_engine.async_generate(
            prompt=prompt, sampling_params=sampling_params
        )
        return res["text"]

    async def explore(self):
        turns: List[Dict[str, Any]] = []
        for i in range(self.max_iterations):
            input_text = self._prepare_input()
            resp = await self.generate(input_text, self.sampling_params)
            answer = self._parse_response(resp)
            self.history.append(answer)
            turns.append(
                {
                    "instance_id": self.instance_id,
                    "trajectory_id": self.trajectory_id,
                    "messages": [
                        {"role": "user", "content": input_text},
                        {"role": "assistant", "content": resp},
                    ],
                    "history": list(self.history),
                }
            )
        return turns


class SimpleExploreAgentGroup:
    # --------- constructor & helpers (unchanged) -----------------------------
    def __init__(
        self,
        batch: DataProto,
        num_trajectories: int,
        infer_engine: Any,
        max_prompt_length: int = 1024,
        max_response_length: int = 1024,
        max_starting_message_length: int = 12000,
        max_parallel_agents: int = 1,
        max_eval_parallel_agents: int = 1,
        max_iterations: int = 10,
        tokenizer: Any = None,
        sampling_params: Any = None,
        device: Any = None,
        remove_think_tokens: bool = False,
        qwen3_enable_thinking: bool = True,
    ) -> None:
        self.batch = batch
        self.num_trajectories = num_trajectories
        self.infer_engine = infer_engine
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.total_len = self.max_prompt_length + self.max_response_length
        self.max_starting_message_length = max_starting_message_length
        self.max_parallel_agents = max_parallel_agents
        self.max_eval_parallel_agents = (
            max_eval_parallel_agents if max_eval_parallel_agents > 0 else max_parallel_agents
        )
        self.max_iterations = max_iterations
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.device = device
        self.qwen3_enable_thinking = qwen3_enable_thinking
        self.remove_think_tokens = remove_think_tokens
        self.agents: Dict[int, Dict[int, SimpleExploreAgent]] = {}
        self.results: Dict[int, Dict[int, Any]] = {}
        self._initialize_agents()

    # ... _initialize_agents, _convert_results_to_dataproto are unchanged ...

    # ------------------------------------------------------------------------
    # NEW, DEAD‑LOCK‑FREE PIPELINE
    # ------------------------------------------------------------------------
    async def generate_trajectories_pipeline(self):  # noqa: C901
        """
        Generate all trajectories with a *fixed* worker pool so we never spawn a
        coroutine after the queue is empty (the cause of the hang you saw).
        """
        total_instances = len(self.batch)
        run_queue: asyncio.Queue[Tuple[int, int]] = asyncio.Queue()

        # -- enqueue every (batch_idx, trajectory_id) exactly once -------------
        for traj in range(self.num_trajectories):
            for bidx in range(total_instances):
                run_queue.put_nowait((bidx, traj))

        # -- worker coroutine --------------------------------------------------
        async def run_one_agent():
            while True:
                try:
                    batch_idx, traj_id = run_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break  # queue drained → exit worker

                instance_id = self.batch[batch_idx].non_tensor_batch["index"]
                try:
                    res = await self._run_agent(batch_idx, traj_id)
                    # store
                    self.results.setdefault(instance_id, {})[traj_id] = res
                finally:
                    run_queue.task_done()

        # -- launch fixed pool -------------------------------------------------
        workers = [
            asyncio.create_task(run_one_agent())
            for _ in range(self.max_parallel_agents)
        ]

        # wait for all queue items to be processed, then wait for workers
        await run_queue.join()
        await asyncio.gather(*workers, return_exceptions=False)

        # convert & return
        return self._convert_results_to_dataproto()

    # ------------------------------------------------------------------------
    def run(self):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.generate_trajectories_pipeline())
