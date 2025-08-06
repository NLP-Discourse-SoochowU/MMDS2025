# Attention: the sft_data_gpt4.json is the same as dpo_data + eval_dpo_data
# ===================================================
# SeaLLM V2
# ===================================================
# SFT
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=1,2 accelerate launch sft_train.py \
    --model_name="ml_models/seallm_7b/7B_V2" \
    --output_dir="ml_models/checkpoints/mmds_v2" \
    --dataset_name="data/tune/tune_mmds.json" \
    --max_steps=1800 \
    --logging_steps=10 \
    --save_steps=10 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --eval_steps=100 \
    --weight_decay=0.05 \
    --optimizer_type="paged_adamw_32bit" \
    --run_name="mmds_v2" \
    --report_to="wandb"