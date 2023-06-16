# Chat Table T5

Chat Table T5 is a finetuned FLAN-T5 model to transform a question related to a specific table into an SQL query that retrieves the answer.

## Overview

This repository is developed with insights from the [self-instruct](https://github.com/yizhongw/self-instruct) and [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca). We tailored the approach to specifically generate question and answer pairs for Text-to-SQL tasks. The main distinctions of this repository include:

1. Our focus lies only on generating question and answer pairs for Text-to-SQL tasks, unlike the aforementioned projects that encompass different domains and tasks.
2. We utilize a SQL_CHECKER (SQLite engine) to filter the training data, ensuring only runnable SQL queries remain.
3. Currently only support querying a single table instead of a complex SQL database.

## Generate training data

Follow the steps below to generate training data:

1. Update prompt tempalte variables in `prompt.py`:

- `table_columns`: Specify the columns of your table.
- `table_name`: Input the name of your table.

2. Modify the `qa_pairs_seed.jsonl`:

This file should include sample questions and target SQL queries related to your table. Ensure that the `table_columns` and `table_name` match those in your sample questions.


3. Set your OpenAI API in your environment:

```bash
export OPENAI_API_BASE="Your OPEN AI BASE"
export OPENAI_API_KEY="YOUR API KEY"
```

4. Execute data generation:

```bash
python generate_instruction.py \
    --output_dir ./data \
    --num_instructions_to_generate 20 \
    --request_batch_size 1 \
    --seed_tasks_path data/qa_pairs_seed.jsonl \
    --csv_file_path "data/mini_price_data.csv" \
    --check_sql \
    --model_name vicuna-7b-v1.1                                  
```

- `check_sql`: This flag indicates whether to check the correctness of the SQL query.
- `num_instructions_to_generate`: Defines the total number of training samples to generate.
- `csv_file_path`: Path to the CSV file used to verify if the generated SQL query is executable. You could check the `data/mini_price_data.csv` as an example.


## Finetuning Model

The finetuning is based on `google/flan-t5-base`:

```bash
python train.py \
    --model_name_or_path google/flan-t5-base \
    --train_data_path ./data/regen.json \
    --bf16 True \
    --output_dir ./output \
    --num_train_epochs 30 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True
```

## Inference

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from prompt import PROMPT_INPUT

model_id = "your_model_output_dir"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

input_text = PROMPT_INPUT.format_map({"question": "how many rows are there in the table?"})

pipe = pipeline(
    "text2text-generation",
    model=model, tokenizer=tokenizer, max_length=512
)
print(pipe(input_text))
```

