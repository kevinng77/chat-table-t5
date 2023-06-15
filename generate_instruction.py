#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#    Copyright 2023 kevin NG
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#    The following instruction data generation code is modified from
#    https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py

import argparse
from typing import List, Union
import time
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool
import numpy as np
import tqdm
from rouge_score import rouge_scorer
from langchain.llms import OpenAI
from langchain.schema import LLMResult
from langchain.llms.base import BaseLLM
import re
from prompt import GEN_DATA_PROMPT_PREFIX, table_name

import pandas as pd
from utils.data_utils import *
from utils.database import init_sql_db
import logging

logging.basicConfig(level=logging.INFO)


def encode_prompt(prompt_instructions: List[dict]):
    """Encode multiple prompt instructions into a single string."""
    prompt = GEN_DATA_PROMPT_PREFIX

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, output) = task_dict["instruction"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def post_process_gpt3_response(
    num_prompt_instructions: int, response: LLMResult, sql_checker: any = None
):
    if response is None:
        return []

    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response.text
    raw_instructions = re.split("###", raw_instructions)

    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and (
            response.generation_info is None
            or response.generation_info["finish_reason"] == "length"
        ):
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Output):", inst)

        if len(splitted_data) != 5:
            continue
        else:
            inst = splitted_data[2].strip()
            output = splitted_data[4].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue

        # TODO filter incorrect SQL query, Customize your sample filter here.
        if sql_checker is not None:
            try:
                sql_checker.run(output)
            except Exception as err:
                logging.warning(err)
                continue

        # If you would like to add additional "input", please refer to alpaca repo:
        # https://github.com/tatsu-lab/stanford_alpaca/blob/main/generate_instruction.py
        instructions.append({"instruction": inst, "input": "", "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def get_sql_checker(csv_file_path=None, check_sql=None):
    """Get a SQL query check, TO check SQL, run `db.run(SQL_QUERY)`"""
    if check_sql is None:
        logging.info(">>> Ignore SQL query check.")
        return None
    logging.info(f">>> Check SQL Query based on file [{csv_file_path}]")
    df = pd.read_csv(csv_file_path)
    sql_engine = init_sql_db(
        table_name=table_name, df=df, database_path="data/sqlite_data.db"
    )
    return sql_engine


def generate_instruction_following_data(
    output_dir: str = "./",
    seed_tasks_path: str = "data/qa_pairs_seed.jsonl",
    num_instructions_to_generate: int = 15,
    num_prompt_instructions: int = 3,
    model: BaseLLM = None,
    request_batch_size: int = 2,
    num_cpus: int = 1,
    csv_file_path: Union[str, None] = None,
    check_sql: bool = False
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["question"], "output": t["output"]} for t in seed_tasks
    ]
    logging.info(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, "regen.json")):
        machine_instruction_data = jload(os.path.join(output_dir, "regen.json"))
        logging.info(
            f"Loaded {len(machine_instruction_data)} machine-generated instructions"
        )

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # first we tokenize all the seed instructions and generated machine instructions
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]

    all_instruction_tokens = [
        scorer._tokenizer.tokenize(inst) for inst in all_instructions
    ]
    sql_checker = get_sql_checker(csv_file_path=csv_file_path, check_sql=check_sql)

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(
                seed_instruction_data, num_prompt_instructions
            )
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)

        request_start = time.time()

        results = model.generate(batch_inputs)
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []

        for result in results.generations:
            new_instructions = post_process_gpt3_response(
                num_prompt_instructions, result[0], sql_checker=sql_checker
            )
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(
                instruction_data_entry["instruction"]
            )
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i]
                for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            instruction_data_entry[
                "most_similar_instructions"
            ] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(
                np.mean(rouge_scores)
            )
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        logging.info(
            f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s"
        )
        logging.info(f"Generated {total} instructions, kept {keep} instructions")

        jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def load_model(
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    batch_size: int,
):
    # Change the LLM you want to used for sample generation. i.e.
    model = OpenAI(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        batch_size=batch_size,
        logit_bias={"50256": -100},
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="localhost")
    parser.add_argument(
        "--seed_tasks_path", type=str, default="data/qa_pairs_seed.jsonl"
    )
    parser.add_argument("--num_instructions_to_generate", type=int, default=20)
    parser.add_argument("--num_prompt_instructions", type=int, default=4)
    parser.add_argument("--request_batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="gpt-4")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--csv_file_path", type=str, default=None)
    parser.add_argument("--check_sql", action="store_true")

    args = parser.parse_args()


    model = load_model(
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.request_batch_size,
    )

    logging.info(f"args: {args}")

    generate_instruction_following_data(
        output_dir=args.output_dir,
        seed_tasks_path=args.seed_tasks_path,
        num_instructions_to_generate=args.num_instructions_to_generate,
        num_prompt_instructions=args.num_prompt_instructions,
        model=model,
        request_batch_size=args.request_batch_size,
        num_cpus=args.num_cpus,
        csv_file_path=args.csv_file_path,
        check_sql=args.check_sql
    )


if __name__ == "__main__":
    main()
