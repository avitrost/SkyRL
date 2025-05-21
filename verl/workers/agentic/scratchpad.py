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
from openhands.controller.agent import Agent
from openhands.core.logger import openhands_logger as logger
from openhands.utils.async_utils import call_sync_from_async, call_async_from_sync
from openhands.llm.llm import LLM
from openhands.core.config import LLMConfig, AgentConfig
from openhands.core.message import Message, TextContent
import re


SCRATCHPAD_TEMPLATE = """
    <prompt>
    {prompt}
    </prompt>

    <step>
    {step}
    </step>

    <previous_attempts>
    {scratchpad}
    </previous_attempts>

    Think step by step to solve this problem.

    Respond in the following format, using careful step-by-step reasoning, and utilizing the following information summarizing previous failed attempts (provided in <previous_attempts>...</previous_attempts> if not the very first try). Once answered, briefly summarize your attempt in the <scratchpad> tag:

    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    <scratchpad>
    ...
    </scratchpad>
    """


class ScratchpadAgent:
    """
    An online implementation of ScratchpadAgent that leverages infer's asynchronous capabilities
    for a single agent instance.
    """
    
    def __init__(
        self,
        instance_id: int,
        trajectory_id: int,
        prompt: str,
        ground_truth: str,
        max_prompt_length: int = 1024,
        infer_engine=None,
        tokenizer=None,
        sampling_params=None,
        max_iterations: int = 10,
        reward_func: Callable = None,
        qwen3_enable_thinking: bool = True,
    ) -> None:
        """
        Initialize a single ScratchpadAgent instance.
        """
        
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.step_count = 0
        self.infer_engine = infer_engine
        self.sampling_params = sampling_params
        
        # Store instance and trajectory IDs separately
        self.instance_id = instance_id
        self.trajectory_id = trajectory_id

        self.max_iterations = max_iterations

        self.prompt = prompt
        self.ground_truth = ground_truth

        self.reward_func = reward_func

        self.qwen3_enable_thinking = qwen3_enable_thinking

    def _initialize_state(self) -> None:
        """Initialize the agent's state."""
        self.state = {
            'instance_id': self.instance_id,
            'trajectory_id': self.trajectory_id,
            'prompt': self.prompt,
            'ground_truth': self.ground_truth,
            'step_count': 0,
            'messages': [],
            'scratchpad': [None] * self.max_iterations,
            'solved_idx': None, # Will be set to index of correct try when solved
            'solved': False,
        }

    def _prepare_input(self) -> torch.Tensor:
        """Prepare input for the model, returning a tensor of token IDs."""

        # Format the template with current state values
        input_text = SCRATCHPAD_TEMPLATE.format(
            prompt=self.state['prompt'],
            step=self.state['step_count'],
            scratchpad=self.state['scratchpad'][-1]  # TODO: make this flexible
        )
        self.state['messages'].append([input_text])

        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        return input_ids
    
    def _parse_response(self, response_str: str) -> Tuple[Optional[str], Optional[str]]:  # TODO: don't hardcode
        # Parse answer
        def parse_answer(response_str: str) -> str:
            """Extract the answer from between <answer> and </answer> tags."""
            answer_match = re.search(r'<answer>(.*?)</answer>', response_str, re.DOTALL)
            if answer_match:
                return answer_match.group(1).strip()
            else:
                return None
        answer = parse_answer(response_str)
        if answer is None:
            logger.warning(f"Failed to parse answer from response: {response_str}")

        # Parse scratchpad info
        def parse_scratchpad_info(response_str: str) -> str:
            """Extract the scratchpad info from between <scratchpad> and </scratchpad> tags."""
            scratchpad_match = re.search(r'<scratchpad>(.*?)</scratchpad>', response_str, re.DOTALL)
            if scratchpad_match:
                return scratchpad_match.group(1).strip()
            else:
                return None
        scratchpad_info = parse_scratchpad_info(response_str)
        if scratchpad_info is None:
            logger.warning(f"Failed to parse scratchpad info from response: {response_str}")

        return answer, scratchpad_info

    def _update_state(self, response_str: str) -> None:
        """Update the agent's state based on the response."""
        self.state['messages'][-1].append(response_str)
        model_answer, scratchpad_info = self._parse_response(response_str)

        self.state['scratchpad'][self.step_count] = scratchpad_info
        ground_truth = self.state['ground_truth']
        is_correct = self.reward_func(model_answer, ground_truth)
        if is_correct:
            self.state['solved_idx'] = self.step_count
            self.state['solved'] = True
    
    async def generate(self, input_ids, sampling_params):
        res = await self.infer_engine.async_generate(input_ids=input_ids, sampling_params=sampling_params)
        response_str = res["text"]
        return response_str

    async def attempt_to_solve(self):
        """Attempt to solve the task."""
        print(f"instance id {self.instance_id}, trajectory {self.trajectory_id}")
        self.step_count = 0
        while self.step_count < self.max_iterations and not self.state["solved"]:
            print(f"step {self.step_count}")
            input_ids = self._prepare_input()
            response_str = call_async_from_sync(self.generate, input_ids=input_ids, sampling_params=self.sampling_params)
            self._update_state(response_str)
            self.step_count += 1

        return_val = {
            'instance_id': self.instance_id,
            'trajectory_id': self.trajectory_id,
            'messages': self.state['messages'],
            'solved_idx': self.state['solved_idx'],
            'solved': self.state['solved'],
        }
        return return_val

class ScratchpadAgentGroup:
    """
    A class that manages multiple ScratchPadAgent instances to generate trajectories in parallel.
    """

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
        qwen3_enable_thinking: bool = True
    ) -> None:
        """
        Initialize the CodeActAgentGroup to manage multiple agent instances.
        
        Args:
            batch: DataProto containing the batch of data
            num_trajectories: Number of trajectories to generate per instance
            infer_engine: The infer engine for generation
            max_prompt_length: Maximum prompt length
            max_parallel_agents: Maximum number of agents to run in parallel
            max_iterations: Maximum number of iterations per agent
            tokenizer: Tokenizer to use for text encoding/decoding
            max_batch_size: Maximum batch size for LLM generation
        """
        self.batch = batch
        self.infer_engine = infer_engine
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.total_len = self.max_prompt_length + self.max_response_length
        # todo: make it a config
        self.max_starting_message_length = max_starting_message_length
        self.max_parallel_agents = max_parallel_agents
        self.max_eval_parallel_agents = max_eval_parallel_agents
        print("max eval parallel agents: ", self.max_eval_parallel_agents)
        if max_eval_parallel_agents <= 0: 
            print(f"`max_eval_parallel_agents` has not been set. Setting it to `max_parallel_agents` i.e {max_parallel_agents}")
            self.max_eval_parallel_agents = max_parallel_agents
        self.max_iterations = max_iterations
        self.num_trajectories = num_trajectories
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.device = device
        
        # Map of instance ID to agent instance
        self.agents = {}
        
        # Map of instance ID to agent results
        self.results = {}
        
        self.qwen3_enable_thinking = qwen3_enable_thinking
        
        # Initialize agents for each instance
        self._initialize_agents()

    def _convert_results_to_dataproto(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Convert the results to a DataProto format.
        
        Returns:
            A dictionary mapping instance ID to a dictionary of trajectory ID to results
        """
        results_dataproto = {}
        for instance_id, trajectories in self.results.items():
            results_dataproto[instance_id] = {}
            for trajectory_id, result in trajectories.items():
                results_dataproto[instance_id][trajectory_id] = result
        print("results dataproto: ", results_dataproto)
        print("PAUSE")
        input("Press Enter to continue...")
        return results_dataproto

    def close(self):
        """Clean up resources"""
            
        # Close all agent instances
        for instance_id in self.agents:
            for trajectory_id in self.agents[instance_id]:
                self._cleanup_agent(instance_id, trajectory_id)
    
    def _cleanup_agent(self, instance_id, trajectory_id):
        try:
            self.agents[instance_id][trajectory_id].close()
        except Exception as e:
            logger.warning(f"Error closing agent {instance_id}, trajectory {trajectory_id}: {str(e)}")

    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        self.close()
    
    def _initialize_agents(self) -> None:
        """Initialize agent instances for each task."""
        for data_item in self.batch:
            instance_id = data_item.non_tensor_batch['index']
            self.agents[instance_id] = {}
            prompt = data_item.non_tensor_batch['raw_prompt'][0]['content']
            ground_truth = data_item.non_tensor_batch['ground_truth']
            for n in range(self.num_trajectories):
                print('************************************')
                print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                print('************************************')
                print(f"Creating agent for instance {instance_id}, trajectory {n}")
                self.agents[instance_id][n] = ScratchpadAgent(
                    instance_id=instance_id,
                    trajectory_id=n,
                    prompt=prompt,
                    ground_truth=ground_truth,
                    max_prompt_length=self.max_prompt_length,
                    tokenizer=self.tokenizer,
                    infer_engine=self.infer_engine,
                    sampling_params=self.sampling_params,
                    qwen3_enable_thinking=self.qwen3_enable_thinking
                )
                # Set the instance data for each agent
                self.agents[instance_id][n].max_iterations = self.max_iterations

    async def _run_agent(self, batch_id: int, trajectory_id: int) -> Dict[str, Any]:
        print("Agent started")
        await asyncio.sleep(1)
        print("Agent done")
        return None

        instance_id = self.batch[batch_id].non_tensor_batch['index']
        """Run a single agent to completion and return the results."""
        agent = self.agents[instance_id][trajectory_id]
        assert agent is not None

        return_val = await call_sync_from_async(ScratchpadAgent.attempt_to_solve, agent)
        return_val =  {
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'messages': return_val['messages'],
                'solved_idx': return_val['solved_idx'],
                'solved': return_val['solved'],
            }
        
        self._cleanup_agent(instance_id, trajectory_id)

        return return_val
    

    async def generate_trajectories_pipeline(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Generate trajectories with pipelined runtime initialization to improve efficiency.
        """
        total_instances = len(self.batch)
        print("Total instances:", total_instances)
        print("*************************************")
        print("*************************************")
        print("*************************************")
        print("*************************************")
        
        # Only need the run queue
        run_queue = asyncio.Queue(maxsize=self.max_parallel_agents)

        print("queue created")
        
        # Fill the run queue
        for trajectory_id in range(self.num_trajectories):
            for batch_idx in range(total_instances):
                await run_queue.put((batch_idx, trajectory_id))

        print("queue filled")
        
        # Track active tasks
        active_run_tasks = set()
        needed_run_tasks = self.num_trajectories * total_instances  # Total tasks we'll eventually need   
        
        # Helper function to run one agent
        async def run_one_agent():
            print("Run one agent started")
            await asyncio.sleep(1)
            print("Run one agent done")
            return None
            print("Waiting for a task to run")
            print("**************************************")
            print("**************************************")
            print("**************************************")
            batch_idx, trajectory_id = await run_queue.get()
            print("Got a task to run")
            instance_id = self.batch[batch_idx].non_tensor_batch['index']
            start_time = time.time()
            try:
                logger.info(f"Running agent for instance {instance_id}, trajectory {trajectory_id}")
                result = await self._run_agent(batch_idx, trajectory_id)
                elapsed = time.time() - start_time
                
                # Store the result
                if instance_id not in self.results:
                    self.results[instance_id] = {}
                self.results[instance_id][trajectory_id] = result
                
                print(f"Successfully completed instance {instance_id}, trajectory {trajectory_id} in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"[This line should not be reached!!] Error running agent for {instance_id}, trajectory {trajectory_id}: {str(e)}")
                # Store error result
                raise e
            finally:
                run_queue.task_done()
                nonlocal needed_run_tasks
                # Start another run task if available
                if needed_run_tasks > 0:
                    needed_run_tasks -= 1
                    task = asyncio.create_task(run_one_agent())
                    active_run_tasks.add(task)
                    task.add_done_callback(lambda t: active_run_tasks.discard(t))
        
        # Start a few agent run tasks (they'll wait on the run_queue)
        for i in range(self.max_parallel_agents):
            print("**************************************")
            print(self.max_parallel_agents)
            assert False
            print(f"Starting initial run task {i}")
            needed_run_tasks -= 1
            task = asyncio.create_task(run_one_agent())
            print(f"Task {i} created")
            active_run_tasks.add(task)
            print(f"Task {i} added to active tasks")
            task.add_done_callback(lambda t: active_run_tasks.discard(t))
            print(f"Task {i} added done callback")
        
        # Wait for all run tasks to complete
        if run_queue.qsize() > 0:
            print(f"Waiting for {run_queue.qsize()} tasks to complete")
            await run_queue.join()
        print("All tasks in queue completed")
        
        # Wait for any remaining active tasks
        all_tasks = active_run_tasks
        if all_tasks:
            logger.info(f"Waiting for {len(all_tasks)} (run: {len(active_run_tasks)}) remaining tasks to complete")
            await asyncio.wait(all_tasks)
        
        print("All tasks completed")
        results_dataproto = self._convert_results_to_dataproto()
        print("Results converted to DataProto format")
        return results_dataproto

    def run(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Run the agent group synchronously by creating a new event loop if necessary.
        
        Returns:
            Dict mapping instance ID to a dict of trajectory ID to results
        """
        print('IN THE RUN FUNCTION')
        print('************************************')
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists in this thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the generate_trajectories coroutine in the event loop
        try:
            return loop.run_until_complete(self.generate_trajectories_pipeline())
        finally:
            # Close the batch manager to ensure cleanup
            self.close()
            # loop.close()