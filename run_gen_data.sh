python generate_instruction.py \
    --output_dir ./ \
    --num_instructions_to_generate 20 \
    --request_batch_size 1 \
    --seed_tasks_path data/qa_pairs_seed.jsonl \
    --database_path "data/your_database.db" \
    --check_sql