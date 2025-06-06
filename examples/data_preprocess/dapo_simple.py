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
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/dapo')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    train_data_source = 'open-r1/DAPO-Math-17k-Processed'
    test_data_source = 'opencompass/AIME2025'

    train_dataset = datasets.load_dataset(train_data_source, 'en')['train']
    test_dataset_1 = datasets.load_dataset(test_data_source, 'AIME2025-I')['test']
    test_dataset_2 = datasets.load_dataset(test_data_source, 'AIME2025-II')['test'].select(range(13)) # To end up with 28 samples for multi-gpu divisibility
    test_dataset = datasets.concatenate_datasets([test_dataset_1, test_dataset_2])

    # instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def train_process_fn(example, idx):
            question_raw = example.pop('prompt')

            question = question_raw

            solution = example.pop('solution')
            data = {
                "data_source": train_data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': solution,
                    "question": question_raw,
                }
            }
            return data
        
        def test_process_fn(example, idx):
            question_raw = example.pop('question')

            question = question_raw

            solution = example.pop('answer')
            data = {
                "data_source": test_data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': solution,
                    "question": question_raw,
                }
            }
            return data

        return train_process_fn if split == 'train' else test_process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
