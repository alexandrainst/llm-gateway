# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
Prepare jsonl with field `input` and `outputs`.
{
    "index" int,
    "input": str,
    "outputs": [str],
}

python prepare.py \
    --save_dir ./ \
    --benchmark synthetic \
    --task niah_single_1 \
    --tokenizer_path tokenizer.model \
    --tokenizer_type nemo \
    --max_seq_length 4096 \
    --model_template_type base \
    --num_samples 10 \
"""
import os
import sys
import argparse
import importlib
import subprocess
import time
import yaml
from pathlib import Path
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='hf', help='[Options] hf, spm, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--num_samples", type=int, default=500, help='maximum number of samples we want to test')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--model_template_type", type=str, default='auto', help='Deprecated: chat templates are applied server-side; kept for compatibility but ignored')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')
parser.add_argument("--prepare_for_ns", action='store_true')

args = parser.parse_args()

def main():
    start_time = time.time()
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        module = importlib.import_module(f"{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.", file=sys.stderr)

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    # Compose plain task template; chat templates are handled by the server.
    task_template = config['template']
    answer_prefix = config.get('answer_prefix', '')
    config['template'] = task_template + answer_prefix

    # Split task into multiple chunks 
    chunks = [(args.num_samples // args.chunk_amount) + (1 if i < args.num_samples % args.chunk_amount else 0) for i in range(args.chunk_amount)]
    num_samples = chunks[args.chunk_idx]
    pre_samples = sum(chunks[:args.chunk_idx])
    
    random_seed = args.random_seed + args.chunk_idx

    
    save_file = args.save_dir / args.task / f"{args.subset}.jsonl"
    file_exists = False
    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            data = f.readlines()
        if len(data) == args.num_samples: file_exists = True



    if not file_exists:
        try:
            script = os.path.join(curr_folder, args.benchmark, f"{config['task']}.py")
            additional_args = " ".join([f"--{k} {v}" for k, v in config['args'].items()])
            python_exec = sys.executable
            command = f"""{python_exec} {script} \
            --save_dir  {args.save_dir} \
            --save_name {args.task} \
            --subset {args.subset} \
            --tokenizer_path {args.tokenizer_path} \
            --tokenizer_type {args.tokenizer_type} \
            --max_seq_length {args.max_seq_length} \
            --tokens_to_generate {config['tokens_to_generate']} \
            --num_samples {num_samples} \
            --random_seed {random_seed} \
            {additional_args} \
            {f"--remove_newline_tab" if args.remove_newline_tab else ""} \
            {f"--pre_samples {pre_samples}" if config['task'] == 'qa' else ""} \
            --template "{config['template']}" \
            """
            # model_template_token is deprecated; leave as 0 if present in scripts.
            
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # Do not print stdout on success to keep parent stdout clean for JSON
            if result.returncode != 0:
                print(result.stderr, file=sys.stderr)
                raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        except subprocess.CalledProcessError as e:
            # Surface only to stderr so upstream JSON stdout remains clean
            print("Error output:", e.stderr, file=sys.stderr)
            # Abort on failure; do not claim success
            raise SystemExit(1)

        # Progress messages to stderr to avoid polluting parent stdout
        print(f"Prepare {args.task} with lines: {args.num_samples} to {save_file}", file=sys.stderr)
        print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes", file=sys.stderr)
    else:
        print(f"Skip preparing {args.task} with lines: {args.num_samples} to {save_file} (file exists)", file=sys.stderr)
    
if __name__ == '__main__':
    main()
