export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

accelerate launch src/sharpdepth/training/train.py \
    --depth_weight 0.4 \
    --base_ckpt_dir jingheya/lotus-depth-g-v1-0 \
    --student_ckpt_dir jingheya/lotus-depth-g-v1-0 \
    --add_datetime_prefix \
    --report_to tensorboard \
    --mixed_precision bf16 \
    --seed 42 \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32 \
    --learning_rate 1e-6 \
    --scale_lr \
    --lr_scheduler cosine \
    --lr_warmup_steps 200 \
    --tracker_project_name main \
    --set_grads_to_none \
    --checkpointing_steps 5000 \
    --validation_steps 2000 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 6 \
    --use_ema

