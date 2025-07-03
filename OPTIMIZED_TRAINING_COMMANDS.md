# Optimized Training Commands for L40S GPU

## Key Optimizations Implemented

### Performance Enhancements:
- **Batch Size**: Increased from 8 to 16 (better GPU utilization)
- **Data Loading**: 8 workers (vs 2) to fix `_prepare_inputs` bottleneck
- **Memory**: Pin memory + persistent workers + prefetch
- **Architecture**: TF32 enabled for Ada architecture speedup
- **Batching**: Group by length to reduce padding waste

### GRPO-Specific Fixes:
- **Completion Length**: 512 tokens (vs 256) to fix 91% truncation issue
- **Generation Batch**: 32 (vs 16) for parallel generation efficiency
- **Effective Batch**: 16×1=16 (still divisible by 8 for GRPO)

## Recommended Training Commands

### Fast Development (0.5B Model)
```bash
uv run train_grpo.py \
    --model_name "./models/Qwen2.5-0.5B-Instruct" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 512 \
    --generation_batch_size 32 \
    --dataloader_num_workers 8 \
    --reward_funcs accuracy format reasoning_steps \
    --no_wandb
```

### Full Training (3B Model) - Recommended
```bash
uv run train_grpo.py \
    --model_name "./models/Qwen2.5-3B-Instruct" \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 512 \
    --generation_batch_size 24 \
    --dataloader_num_workers 8 \
    --reward_funcs accuracy format reasoning_steps cosine repetition_penalty \
    --wandb_project "deepseek-r1-zero-grpo-optimized"
```

### Conservative (if memory issues)
```bash
uv run train_grpo.py \
    --model_name "./models/Qwen2.5-0.5B-Instruct" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_completion_length 512 \
    --generation_batch_size 16 \
    --dataloader_num_workers 6 \
    --reward_funcs accuracy format reasoning_steps \
    --no_wandb
```

## Expected Performance Improvements

### Speed Improvements:
- **Data Loading**: 50-70% faster (8 workers + prefetch + pin memory)
- **GPU Utilization**: 19% → 70-90% (better batching + TF32)
- **Generation**: 30-50% faster (larger generation batches)
- **Overall**: 2-4x faster training steps

### Quality Improvements:
- **Completion Truncation**: 91% → <10% (512 vs 256 tokens)
- **Format Compliance**: Expected improvement from 32% → 60%+
- **Reasoning Quality**: Better due to complete reasoning chains

## Monitoring Commands

### Check GPU Utilization:
```bash
watch -n 1 nvidia-smi
```

### Monitor Training Progress:
```bash
tail -f training.log
```

### Check Process Performance:
```bash
top -p $(pgrep -f train_grpo.py)
```

## Troubleshooting

### If OOM Error:
1. Reduce `per_device_train_batch_size` to 12 or 8
2. Reduce `generation_batch_size` to 16
3. Enable gradient checkpointing (in config)

### If Still Slow:
1. Check `nvidia-smi` for GPU utilization
2. Verify `dataloader_num_workers` is working
3. Monitor `_prepare_inputs` time in logs

### If Format Issues Persist:
1. Increase format reward weight
2. Check completion examples in logs
3. Verify 512 token limit is sufficient