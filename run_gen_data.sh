python generate_instruction.py \
    --output_dir ./data \
    --num_instructions_to_generate 20 \
    --request_batch_size 1 \
    --seed_tasks_path data/qa_pairs_seed.jsonl \
    --csv_file_path "data/mini_price_data.csv" \
    --check_sql \
    --model_name vicuna-7b-v1.1