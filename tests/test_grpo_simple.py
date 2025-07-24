from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
dataset = load_dataset("trl-lib/tldr", split="train")


def reward_func(completions, **kwargs):
    # Dummy reward function that rewards completions with more unique letters.
    return [float(len(set(completion))) for completion in completions]


model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B-Instruct',
        attn_implementation="flash_attention_2",
        use_cache=True,
        torch_dtype="bfloat16",
        device_map="cuda:0"
    )
# tokenizer = AutoTokenizer.from_pretrained(
#     'Qwen/Qwen2.5-0.5B-Instruct',
#     revision="main",
#     trust_remote_code=True,
#     )
# tokenizer.pad_token = tokenizer.eos_token

# tokenizer.padding_side = "left"

config = GRPOConfig(
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,
    use_vllm=True,
    use_liger_loss=False,
)

if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "12355"

trainer = GRPOTrainer(
    model=model,
    # processing_class=tokenizer,
    reward_funcs=reward_func,
    train_dataset=dataset,
    args=config,
)

print("start training")
print("="*100)



trainer.train()