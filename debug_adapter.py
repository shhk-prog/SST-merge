import torch
import sys

# 保存されたアダプターを読み込み
adapter_path = "saved_adapters/llama-3.1-8b/sst_merge/harmful_adapter_1.pt"
print(f"Loading adapter from: {adapter_path}")

adapter_data = torch.load(adapter_path)

print("\nTop-level keys:")
for key in adapter_data.keys():
    print(f"  - {key}")

if 'adapter' in adapter_data:
    print("\nAdapter keys (first 5):")
    adapter_keys = list(adapter_data['adapter'].keys())
    for i, key in enumerate(adapter_keys[:5]):
        print(f"  {i+1}. {key}")
    print(f"\nTotal adapter parameters: {len(adapter_keys)}")
else:
    print("\nNo 'adapter' key found. Direct keys (first 5):")
    keys = list(adapter_data.keys())
    for i, key in enumerate(keys[:5]):
        if key != 'metadata':
            print(f"  {i+1}. {key}")
