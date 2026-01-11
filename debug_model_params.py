import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# モデルを読み込み
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Loading model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# LoRA設定
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none"
)

# PeftModelを作成
peft_model = get_peft_model(model, lora_config)

print("\nModel parameter names (first 10 LoRA params):")
count = 0
for name, param in peft_model.named_parameters():
    if 'lora' in name.lower():
        print(f"  {count+1}. {name}")
        count += 1
        if count >= 10:
            break

print(f"\nTotal LoRA parameters: {count}")

# 保存されたアダプターのキーと比較
print("\n" + "="*80)
print("Saved adapter keys (from debug_adapter.py):")
print("  1. base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight")
print("  2. base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight")
