# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.utils.reward_score.explore import is_equiv, is_correct
import torch
from collections import defaultdict

def explore_compute_score(history):
    """
    Returns 1 if the last response in the history is not equivalent to any previous responses,
    otherwise returns 0.
    This function is used to count the number of distinct answers given for exploration tasks.
    """
    answer = history[-1]
    if answer is None or answer == "":
        return {"score": 0, "explore_score": 0}
    for past_response in history[:-1]:
        if is_equiv(answer, past_response):
            return {"score": 0, "explore_score": 0}
    return {"score": 1, "explore_score": 1}

def pass_at_k_compute_score(history, is_last_turn, ground_truth, explore_only):
    """
    Computes the pass@k score based on the history and ground truth.
    Returns 1 if the last response in the history is equivalent to the ground truth,
    otherwise returns 0.
    """
    if not is_last_turn:
        return {"score": 0, "pass@k_score": 0}
    for past_response in history:
        if is_correct(past_response, ground_truth):
            k = len(history)
            return {"score": k if not explore_only else 0, "pass@k_score": k}  # we still log the metric even if explore_only is True
    return {"score": 0, "pass@k_score": 0}

def compute_score(history, is_last_turn=False, ground_truth=None, explore_only=False):
    """
    Computes the score for the given history.
    If explore_only is True, it only computes the exploration score.
    """
    result1 = explore_compute_score(history)
    result2 = pass_at_k_compute_score(history, is_last_turn, ground_truth, explore_only)
    # Combine the scores from both functions
    combined_score = {
        "score": result1["score"] + result2["score"],
        "explore_score": result1["explore_score"],
        "pass@k_score": result2["pass@k_score"]
    }
    return combined_score


class ExploreRewardManager:
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 reward_fn_key='data_source',
                 use_parallel=False,
                 config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = explore_compute_score
        self.reward_fn_key = reward_fn_key
        self.use_parallel = use_parallel
        self.explore_only = config.reward_model.get("explore_only", False)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        import concurrent.futures

        def compute_score_for_item(i, data_item):
            history = data_item.non_tensor_batch['history']
            is_last_turn = data_item.non_tensor_batch['is_last_turn']

            # The input_ids contains both prompt and response concatenated
            # We need to separate them using the response tensor
            full_input_ids = data_item.batch['input_ids']
            response_ids = data_item.batch['responses']
            full_attention_mask = data_item.batch['attention_mask']
            
            # Calculate the actual prompt length
            response_length = response_ids.shape[-1]
            prompt_length = full_input_ids.shape[-1] - response_length
            
            # Extract prompt part from the concatenated tensor
            prompt_ids = full_input_ids[:prompt_length]
            prompt_attention_mask = full_attention_mask[:prompt_length]
            response_attention_mask = full_attention_mask[prompt_length:]
            
            # Get valid lengths and extract valid tokens
            valid_prompt_length = prompt_attention_mask.sum()
            valid_response_length = response_attention_mask.sum()
            
            # For left-padded prompts, take the last valid_prompt_length tokens
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            # For right-padded responses, take the first valid_response_length tokens  
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # print(f"naive reward manager {self.tokenizer=}")
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            response_ids = data.batch['responses']
            response_length = response_ids.shape[-1]

            valid_response_length = data.batch['attention_mask'][:, -response_length:].sum(-1)

            # extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # print(f"naive reward manager {self.tokenizer=}")
            score = self.compute_score(history=history, is_last_turn=is_last_turn, ground_truth=ground_truth, explore_only=self.explore_only)

            return i, valid_response_length, score, data_source, prompt_str, response_str, ground_truth

        if self.use_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(data)) as executor:
                futures = [executor.submit(compute_score_for_item, i, data[i]) for i in range(len(data))]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            results = [compute_score_for_item(i, data[i]) for i in range(len(data))]

        for i, valid_response_length, score, data_source, prompt_str, response_str, ground_truth in results:
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        reward_tensor_dict = {"all": reward_tensor}

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor_dict, {}
