python train.py \
    --run_name flan-t5-base\
    --model_name_or_path google/flan-t5-base \
    --overwrite_output_dir \
    --train_data_path ./data/regen.json \
    --bf16 True \
    --output_dir ./output \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True