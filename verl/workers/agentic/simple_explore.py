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

    def _convert_results_to_dataproto(self) -> DataProto:
        """
        Convert results to DataProto format for training.
        
        Args:
            results: Dictionary of results, with structure {instance_id: {trajectory_id: result_dict}}
            input_dataproto: The input DataProto that contains the original batch data
            tokenizer: The tokenizer to use for encoding messages
            
        Returns:
            DataProto: A DataProto object with the converted results
        """

        # Non-tensor data
        history_list = []
        
        # Create a mapping of instance_id -> list of trajectories
        instance_trajectories = {}
        for instance_id, trajectories in self.results.items():
            instance_trajectories[instance_id] = []
            for trajectory_id, result in trajectories.items():
                instance_trajectories[instance_id].append(result)

        # Create the final results in the same order as the batch
        matched_results = []
        # instance_list = []
        for batch_item in self.batch:
            instance_id = batch_item.non_tensor_batch['index']
            # instance = batch_item.non_tensor_batch['instance']
            if instance_id in instance_trajectories:
                # Add all trajectories for this instance
                traj_results = instance_trajectories[instance_id]
                matched_results.extend(traj_results)
                # instance_list.extend([instance] * len(traj_results))
        
        assert len(matched_results) == self.num_trajectories * len(self.batch), f"Expected number of results {self.num_trajectories * len(self.batch)}, got {len(matched_results)}"
        
        # # Group results by instance_id for message handling
        # results_by_instance = {}
        # for i, result in enumerate(matched_results):
        #     instance_id = instance_list[i]['instance_id']
        #     if instance_id not in results_by_instance:
        #         results_by_instance[instance_id] = []
        #     results_by_instance[instance_id].append((i, result))
        
        # # Handle empty messages by copying from another trajectory of the same instance
        # for instance_id, results in results_by_instance.items():
        #     # Find a valid messages list to use as fallback
        #     valid_messages = None
        #     valid_patch = None
        #     for _, result in results:
        #         messages = result.get('messages', [])
        #         if messages and len(messages) > 0:
        #             valid_messages = messages
        #             valid_patch = result.get('git_patch', None)
        #             valid_resolved = result.get('resolved', False)
        #             valid_finish = result.get('finish', False)
        #             valid_error = result.get('error', None)
        #             break
            
        #     # If we found valid messages, use them for trajectories with empty messages
        #     if valid_messages:
        #         for idx, result in results:
        #             if not result.get('messages') or len(result.get('messages', [])) == 0:
        #                 print(f"Got empty messages for instance_id {instance_id}, trajectory {idx}. Copying messages array from a valid trajectory. ")
        #                 # Copy messages from the valid trajectory
        #                 matched_results[idx]['messages'] = valid_messages.copy()
        #                 matched_results[idx]['git_patch'] = valid_patch
        #                 matched_results[idx]['resolved'] = valid_resolved
        #                 matched_results[idx]['error'] = valid_error
        #                 matched_results[idx]['finish'] = valid_finish
        
        # Get batch of messages
        all_messages = []
        all_prompts = []
        all_responses = []
        # TODO: iterate through each inner list
        for result_series in matched_results:
            for result in result_series:
                messages = result.get('messages', [])
                all_messages.append(messages)
                # get the response: starting from the first assistant message
                starting_index = 0
                for i, msg in enumerate(messages):
                    if msg["role"] == 'assistant':
                        starting_index = i
                        break
                if starting_index == 0:
                    # If we don't find an assistant, all messages are prompts and there are no responses
                    print(f'ERROR: Found no assistant message. len(messages) == {len(messages)} and roles are {[msg["role"] for msg in messages]}')
                    starting_index = len(messages)
                prompt = messages[:starting_index]
                all_prompts.append(prompt)
                response = messages[starting_index:]
                all_responses.append(response)

                # Also add non-tensor data
                history_list.append(result.get('history', None))


        # Encode messages, get assitant mask and position ids
        prompt_encodings = self.tokenizer.apply_chat_template(
            all_prompts, 
            # return_tensors="pt",
            add_generation_prompt=False,
            return_dict=True,
            padding=True
        )
        prompt_input_ids = torch.tensor(prompt_encodings['input_ids'], device=self.device)
        prompt_attention_mask = torch.tensor(prompt_encodings['attention_mask'], device=self.device)
        prompt_input_ids, prompt_attention_mask = convert_right_padding_to_left(self.tokenizer, prompt_input_ids, prompt_attention_mask, self.device, self.max_starting_message_length)

        response_encodings = self.tokenizer.apply_chat_template(
            all_responses,
            chat_template=chat_template_qwen3_thinking if self.remove_think_tokens else chat_template,
            # return_tensors="pt",
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
            return_dict=True,
            padding=True
        )
        
        response_ids, response_attention_mask, response_assistant_mask = pad_to_max_length_right(
            self.tokenizer, response_encodings, self.total_len, self.device)
            
        
        input_ids = torch.cat([prompt_input_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Create tensor dictionary
        logger.info(f"input_ids shape: {input_ids.shape}, response_ids shape: {response_ids.shape}, max_starting_message_length: {self.max_starting_message_length}, max_response_length: {self.total_len}")
        assert input_ids.shape[1] == attention_mask.shape[1] == position_ids.shape[1], f"input_ids shape {input_ids.shape}, attention_mask shape {attention_mask.shape}, position_ids shape {position_ids.shape} do not match"
        assert response_ids.shape[1] == response_assistant_mask.shape[1], f"response_ids shape {response_ids.shape}, response_assistant_mask shape {response_assistant_mask.shape} do not match"
        tensor_dict = {
            'input_ids': input_ids,
            'responses': response_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': response_assistant_mask,
        }

        # Create non-tensor dictionary
        non_tensor_dict = {
            'history': history_list,
        }
        
        # Create and return DataProto
        result_dataproto = DataProto.from_dict(
            tensors=tensor_dict,
            non_tensors=non_tensor_dict
        )
        
        return result_dataproto
    
    def _initialize_agents(self) -> None:
        """Initialize agent instances for each task."""
        for data_item in self.batch:
            instance_id = data_item.non_tensor_batch['index']
            self.agents[instance_id] = {}
            prompt = data_item.non_tensor_batch['raw_prompt'][0]['content']
            for n in range(self.num_trajectories):
                print('************************************')
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                print('************************************')
                print(f"Creating agent for instance {instance_id}, trajectory {n}")
                self.agents[instance_id][n] = SimpleExploreAgent(
                    instance_id=instance_id,
                    trajectory_id=n,
                    prompt=prompt,
                    max_prompt_length=self.max_prompt_length,
                    tokenizer=self.tokenizer,
                    infer_engine=self.infer_engine,
                    sampling_params=self.sampling_params,
                    qwen3_enable_thinking=self.qwen3_enable_thinking
                )
                # Set the instance data for each agent
                self.agents[instance_id][n].max_iterations = self.max_iterations


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
