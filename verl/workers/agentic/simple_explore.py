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


SIMPLE_EXPLORE_TEMPLATE = """
    <prompt>
    {prompt}
    </prompt>

    <history>
    {history}
    </history>
    
    Respond in the following format, using careful step-by-step reasoning.
    The <history> tag contains the history of previous attempted answers, which you can use to inform your reasoning.
    Inside the <answer> tag, put your final answer as LaTeX inside `$...$` formatting.
    E.g. <answer>$17$</answer>.
    It is critical that the answer provided in the <answer> tag is DIFFERENT from the previous attempts, even if it may be wrong. AN answer must be provided regardless.

    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """

# this is for the tokenizer.apply_chat_template to be able to generate assistant masks directly
# todo: this is a hack, we should find a better way to do this
chat_template = (
        "{% for message in messages %}"
        "{% if (message['role'] != 'assistant') %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}"
        "{{message['content'] + '<|im_end|>'}}"
        "{% endgeneration %}"
        "{{'\n'}}"
        "{% endif %}"
        "{% endfor %}"
    )

# chat template for qwen3 thinking mode to remove think tokens similar to generation phase
chat_template_qwen3_thinking = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% generation %}"
    "{% set full_content = message['content'] %}"
    "{% set mycontent = message['content'] %}"
    "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
    "{% if '</think>' in full_content and not is_last_message %}"
    "{% set mycontent = full_content.split('</think>')[-1].lstrip('\n') %}"
    "{% endif %}"
    "{{mycontent + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\n'}}"
    "{% endif %}"
    "{% endfor %}"
)

    
def convert_right_padding_to_left(tokenizer, input_ids, attention_mask, device, max_len=None):
    """
    Converts right-padded tensors to left-padded tensors with optional custom length.
    
    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        input_ids (torch.Tensor): Right-padded input IDs tensor of shape [batch_size, seq_length]
        attention_mask (torch.Tensor): Right-padded attention mask tensor of shape [batch_size, seq_length]
        device: The device to place the new tensors on
        max_len (int, optional): The desired maximum length of the returned tensors.
                                If None, uses the original sequence length.
    
    Returns:
        tuple: (left_padded_input_ids, left_padded_attention_mask)
    """
    batch_size, orig_seq_length = input_ids.size()
    
    # Use original length if max_len is not specified
    seq_length = max_len if max_len is not None else orig_seq_length
    
    # Create new tensors with the desired size
    left_padded_input_ids = torch.full((batch_size, seq_length), 
                                     tokenizer.pad_token_id, 
                                     dtype=input_ids.dtype, 
                                     device=device)
    left_padded_attention_mask = torch.zeros((batch_size, seq_length), 
                                           dtype=attention_mask.dtype, 
                                           device=device)
    
    for i in range(batch_size):
        # Get the non-padded length of this sequence
        seq_len = attention_mask[i].sum().item()
        
        # Trim sequence if it's longer than max_len
        if seq_len > seq_length:
            logger.warning(f"Trimming sequence length from {seq_len} to {seq_length}")
            seq_len = seq_length
        
        # Calculate the offset for left padding
        offset = seq_length - seq_len
        
        # Copy the non-padded tokens to the end
        left_padded_input_ids[i, offset:] = input_ids[i, :seq_len]
        left_padded_attention_mask[i, offset:] = 1  # Set attention mask for non-padding tokens
    
    return left_padded_input_ids, left_padded_attention_mask

def pad_to_max_length_right(tokenizer, encodings, max_length, device):
    """
    Pads tokenizer outputs to a specific maximum length with configurable padding side.
    
    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        encodings (dict): Dictionary containing 'input_ids', 'attention_mask', and optionally 'assistant_masks'
        max_length (int): The desired maximum length to pad to
        device: The device to place the tensors on
        
    Returns:
        dict: Dictionary with padded tensors for 'input_ids', 'attention_mask', and 'assistant_masks' if present
    """
    batch_size = len(encodings['input_ids'])
    
    # Initialize output tensors
    padded_input_ids = torch.full((batch_size, max_length), 
                                tokenizer.pad_token_id, 
                                dtype=torch.long, 
                                device=device)
    padded_attention_mask = torch.zeros((batch_size, max_length), 
                                      dtype=torch.long, 
                                      device=device)
    padded_assistant_mask = torch.zeros((batch_size, max_length), 
                                          dtype=torch.long, 
                                          device=device)
    
    # Fill tensors with actual values
    num_trimmed = 0
    for i in range(batch_size):
        seq_len = encodings["attention_mask"][i].sum().item() if isinstance(encodings["attention_mask"][i], torch.Tensor) else sum(encodings["attention_mask"][i])
        # Trim if longer than max_length
        actual_len = min(seq_len, max_length)
        if seq_len > max_length:
            logger.warning(
                f"Trimming sequence length from {seq_len} to {actual_len} for batch item {i}"
            )
            num_trimmed += 1
        
        # Right padding - copy sequence data to the beginning
        padded_input_ids[i, :actual_len] = torch.tensor(encodings['input_ids'][i][:actual_len], device=device)
        padded_attention_mask[i, :actual_len] = torch.tensor(encodings['attention_mask'][i][:actual_len], device=device)
        padded_assistant_mask[i, :actual_len] = torch.tensor(encodings['assistant_masks'][i][:actual_len], device=device)
    
    logger.info(f"Trimmed {num_trimmed*100 / max(batch_size, 1)}% of samples in the batch of size {batch_size}")
    return padded_input_ids, padded_attention_mask, padded_assistant_mask


class SimpleExploreAgent:
    """
    An agent that leverages infer's asynchronous capabilities
    for a single agent instance.
    """
    
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
        """
        Initialize a single SimpleExploreAgent instance.
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

        self.reward_func = reward_func

        self.qwen3_enable_thinking = qwen3_enable_thinking

        self.history = []  # Initialize history as an empty list

    def _prepare_input(self) -> tuple[str, str]:
        """Prepare input for the model, returning both user content and chat-templated input text.
        
        Returns:
            tuple: (user_content, templated_input_text)
        """
        # Format the template with current state values
        formatted_history = "\n".join(
            [f"Turn {i}: {answer}" for i, answer in enumerate(self.history)]
        )
        user_content = SIMPLE_EXPLORE_TEMPLATE.format(
            prompt=self.prompt,
            history=formatted_history
        )
        
        # Create messages in chat format
        messages = [
            {
                'role': 'user',
                'content': user_content,
            }
        ]
        
        # Apply chat template to get properly formatted prompt for LLM
        # Note: We use tokenize=False here because we need text input for the LLM.
        # The _convert_results_to_dataproto method will apply chat template with 
        # tokenization and padding for training data preparation.
        templated_input_text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True,
            tokenize=False
        )

        return user_content, templated_input_text
    
    def _parse_response(self, response_str: str) -> Optional[str]:  # TODO: don't hardcode
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
            # print(f"Failed to parse answer from response")
            # logger.warning(f"Failed to parse answer from response: {response_str}")
            pass

        return answer
    
    async def generate(self, prompt, sampling_params):
        # print(f"Generating response for instance {self.instance_id}, trajectory {self.trajectory_id}")
        res = await self.infer_engine.async_generate(prompt=prompt, sampling_params=sampling_params)
        response_str = res["text"]
        # print(f"Generated response for instance {self.instance_id}, trajectory {self.trajectory_id}")
        return response_str
    
    async def explore(self, turn: int) -> Dict[str, Any]:
        # print(f"Exploring instance {self.instance_id}, trajectory {self.trajectory_id}, iteration {turn}/{self.max_iterations}")
        # Generate the next step
        
        # Get both user content and chat-templated input
        user_content, templated_input_text = self._prepare_input()
        
        # print(f"Input text for generation {turn}:", templated_input_text)
        response_str = await self.generate(prompt=templated_input_text, sampling_params=self.sampling_params)
        # response_str = call_async_from_sync(self.generate, prompt=templated_input_text, sampling_params=self.sampling_params)
        # print("GOT A RESPONSE STRING")
        extracted_answer = self._parse_response(response_str)
        self.history.append(extracted_answer)  # update history
        
        # Construct messages with the original user content and LLM response
        messages = [
            {
                'role': 'user',
                'content': user_content,  # Store original user content without chat template
            },
            {
                'role': 'assistant',
                'content': response_str,  # LLM response (already includes chat template formatting)
            }
        ]
        turn_return_val = {
            'instance_id': self.instance_id,
            'trajectory_id': self.trajectory_id,
            'messages': messages,
            'history': self.history.copy(),  # copy the history to avoid mutation
            'is_last_turn': turn == self.max_iterations - 1,
        }
        return turn_return_val

class SimpleExploreAgentGroup:
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
        remove_think_tokens: bool = False,
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

        self.remove_think_tokens = remove_think_tokens
        if self.remove_think_tokens:
            logger.info("Removing think tokens....")
    
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
        all_trajectory_ids = []
        all_instance_ids = []
        all_is_last_turns = []
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
                all_trajectory_ids.append(result.get('trajectory_id', None))
                all_instance_ids.append(result.get('instance_id', None))
                all_is_last_turns.append(result.get('is_last_turn', False))


        # Encode messages, get assistant mask and position ids
        # Note: Here we tokenize and pad for training data preparation.
        # This is different from _prepare_input() which only generates text for LLM.
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

        user_content = [prompt[0]['content'] for prompt in all_prompts]

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
            'trajectory_ids': all_trajectory_ids,
            'instance_ids': all_instance_ids,
            'user_content': user_content,
            'is_last_turn': all_is_last_turns,
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
            prompt = data_item.non_tensor_batch['raw_prompt'][0]['content'] # Raw prompt does not include the chat template
            for n in range(self.num_trajectories):
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

    async def _run_agent(self, batch_id: int, trajectory_id: int, turn: int) -> Dict[str, Any]:
        instance_id = self.batch[batch_id].non_tensor_batch['index']
        """Run a single agent to completion and return the results."""
        agent = self.agents[instance_id][trajectory_id]
        assert agent is not None

        # return_vals = await call_sync_from_async(SimpleExploreAgent.explore, agent)
        return_val = await agent.explore(turn)
        # print(f"Return values: {return_vals}")

        self.results[instance_id][trajectory_id].append(return_val)
    

    async def generate_trajectories_pipeline(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Generate trajectories with pipelined runtime initialization to improve efficiency.
        """
        total_instances = len(self.batch)
        
        # Only need the run queue
        # run_queue = asyncio.Queue(maxsize=self.max_parallel_agents)

        # Initialize results dictionary
        for batch_idx in range(total_instances):
            instance_id = self.batch[batch_idx].non_tensor_batch['index']
            self.results[instance_id] = {}
            for trajectory_id in range(self.num_trajectories):
                self.results[instance_id][trajectory_id] = []

        for turn in range(self.max_iterations):
            async with asyncio.TaskGroup() as tg:
                for batch_idx in range(total_instances):
                    for trajectory_id in range(self.num_trajectories):
                        # Create a task for each agent
                        tg.create_task(self._run_agent(batch_idx, trajectory_id, turn))
                        
        results_dataproto = self._convert_results_to_dataproto()
        return results_dataproto

    def run(self) -> Dict[int, Dict[int, Dict[str, Any]]]:
        """
        Run the agent group synchronously by creating a new event loop if necessary.
        
        Returns:
            Dict mapping instance ID to a dict of trajectory ID to results
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop exists in this thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the generate_trajectories coroutine in the event loop
        return loop.run_until_complete(self.generate_trajectories_pipeline())
