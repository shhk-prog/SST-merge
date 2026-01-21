"""
データフリーSST-Merge実行スクリプト

既存LoRAアダプターを読み込み、学習データなしでマージを実行
"""

import torch
from pathlib import Path
import logging
from sst_merge_data_free import SSTMergeDataFree, save_merged_adapter
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#####################################################
# 設定
#####################################################

# ベースモデル
model_id = "meta-llama/Llama-3.1-8B-Instruct"

# マージするアダプターのペア
merge_pairs = [
    # (Utility adapter, Safety adapter, output_name)
    ("./FT_model/A5_utility_meta_llama_3.1_8b_instruct_repliqa_r16_10ep_lr2e-4",
     "./FT_model/A7_safety_meta_llama_3.1_8b_instruct_r16_7ep_lr2e-4",
     "A5_A7"),
    
    ("./FT_model/A6_utility_meta_llama_3.1_8b_instruct_alpaca_r16_7ep_lr2e-4",
     "./FT_model/A7_safety_meta_llama_3.1_8b_instruct_r16_7ep_lr2e-4",
     "A6_A7"),
]

# SST-Merge設定
k_values = [5,10,20]  # k値（データフリー版では命名用、将来的にtop-k ratioとして使用可能）
use_layerwise_options = [True,False]  # Layerwise重み調整: [False] or [True] or [False, True]
alpha_values = [0.05,0.07,0.09,0.1,0.12,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
use_gevp = True  # GEVP使用（Falseの場合はシンプルなTask Arithmetic風マージ）
top_k_ratio_options = [None]  # [None]=ソフトマスク, [0.3]=上位30%, 複数指定可能

# 出力ディレクトリ
output_dir = "./merge_model_data_free"

#####################################################


def convert_adapter_to_full(
    adapter_path: str,
    output_path: str,
    base_model_id: str
) -> bool:
    """
    アダプターをフルモデルに変換
    
    Args:
        adapter_path: アダプターディレクトリパス
        output_path: 出力ディレクトリパス
        base_model_id: ベースモデルID
    
    Returns:
        成功したらTrue
    """
    try:
        logger.info(f"Loading base model: {base_model_id}")
        
        # ベースモデルをロード
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map='auto'
        )
        
        logger.info(f"Loading adapter from: {adapter_path}")
        
        # アダプターをロード
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        logger.info("Merging adapter into base model...")
        
        # アダプターをベースモデルにマージ
        model = model.merge_and_unload()
        
        # トークナイザーもロード
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        
        # 保存
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving full model to: {output_path}")
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"✓ Full model saved successfully")
        
        # メモリ解放
        del model
        del base_model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert adapter to full model: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_adapter_weights(adapter_path: str) -> dict:
    """
    LoRAアダプターの重みを読み込み
    
    Args:
        adapter_path: アダプターディレクトリパス
    
    Returns:
        adapter_dict: LoRA重み辞書
    """
    from safetensors.torch import load_file
    
    adapter_dir = Path(adapter_path)
    
    # .binファイルを優先、なければ.safetensorsを読み込む
    adapter_bin = adapter_dir / "adapter_model.bin"
    adapter_safetensors = adapter_dir / "adapter_model.safetensors"
    
    if adapter_bin.exists():
        adapter_dict = torch.load(adapter_bin, map_location='cpu')
        logger.info(f"Loaded adapter from .bin: {adapter_path}")
    elif adapter_safetensors.exists():
        adapter_dict = load_file(str(adapter_safetensors))
        logger.info(f"Loaded adapter from .safetensors: {adapter_path}")
    else:
        raise FileNotFoundError(
            f"Adapter not found: neither {adapter_bin} nor {adapter_safetensors} exists"
        )
    
    logger.info(f"  Keys: {len(adapter_dict)}")
    
    return adapter_dict


def merge_and_save_full_model(
    base_model_id: str,
    merged_adapter: dict,
    output_path: str
):
    """
    マージされたアダプターをベースモデルに統合して保存
    
    Args:
        base_model_id: ベースモデルID
        merged_adapter: マージされたアダプター
        output_path: 出力パス
    """
    logger.info(f"\nLoading base model: {base_model_id}")
    
    # ベースモデルとトークナイザーを読み込み
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # PEFTモデルとして保存（adapter_model.binとして）
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(merged_adapter, output_path / "adapter_model.bin")
    
    # adapter_config.jsonも必要（既存アダプターからコピー）
    # ここでは簡略化のためスキップ
    
    logger.info(f"Merged adapter saved to: {output_path}")


def main():
    logger.info("="*70)
    logger.info("Data-Free SST-Merge Execution")
    logger.info("="*70)
    
    # 統計情報
    total_tasks = len(merge_pairs) * len(k_values) * len(use_layerwise_options) * len(top_k_ratio_options) * len(alpha_values)
    completed = 0
    
    logger.info(f"\nTotal configurations: {total_tasks}")
    logger.info(f"  Pairs: {len(merge_pairs)}")
    logger.info(f"  k values: {k_values}")
    logger.info(f"  Layerwise options: {use_layerwise_options}")
    logger.info(f"  Top-k ratio options: {top_k_ratio_options}")
    logger.info(f"  Alpha values: {len(alpha_values)}")
    
    for utility_path, safety_path, pair_name in merge_pairs:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing pair: {pair_name}")
        logger.info(f"  Utility: {utility_path}")
        logger.info(f"  Safety: {safety_path}")
        logger.info(f"{'='*70}")
        
        # Step 1: アダプター読み込み
        try:
            utility_adapter = load_adapter_weights(utility_path)
            safety_adapter = load_adapter_weights(safety_path)
        except FileNotFoundError as e:
            logger.error(f"Skipping {pair_name}: {e}")
            continue
        
        # Step 2: 各設定の組み合わせでマージ
        for k in k_values:
            for use_layerwise in use_layerwise_options:
                for top_k_ratio in top_k_ratio_options:
                    for alpha in alpha_values:
                        logger.info(f"\n{'-'*60}")
                        logger.info(f"k={k}, layerwise={use_layerwise}, top_k={top_k_ratio}, α={alpha}")
                        logger.info(f"{'-'*60}")
                        
                        # SST-Merge実行
                        sst_merge = SSTMergeDataFree(
                            safety_weight=alpha,
                            use_gevp=use_gevp,
                            top_k_ratio=top_k_ratio,
                            use_layerwise=use_layerwise  # layerwiseオプション追加
                        )
                        
                        merged_adapter = sst_merge.merge(utility_adapter, safety_adapter)
                        
                        # 出力パス生成
                        output_name = f"{pair_name}_data_free_k{k}"
                        if use_layerwise:
                            output_name += "_lw"
                        output_name += f"_a{alpha}"
                        if top_k_ratio is not None:
                            output_name += f"_topk{int(top_k_ratio*100)}"
                        
                        adapter_output = Path(output_dir) / output_name
                        full_output = Path(output_dir + "_full") / output_name
                        
                        # Step 1: アダプターを保存
                        save_merged_adapter(merged_adapter, adapter_output, utility_path)
                        logger.info(f"✓ Adapter saved: {output_name}")
                        
                        # Step 2: フルモデルに変換
                        logger.info(f"Converting adapter to full model: {output_name}")
                        success = convert_adapter_to_full(
                            str(adapter_output),
                            str(full_output),
                            model_id
                        )
                        
                        if success:
                            logger.info(f"✓ Full model saved: {output_name}")
                            completed += 1
                        else:
                            logger.error(f"✗ Full model conversion failed: {output_name}")
    
    logger.info("\n" + "="*70)
    logger.info(f"Data-Free SST-Merge completed!")
    logger.info(f"  Total configurations: {total_tasks}")
    logger.info(f"  Completed: {completed}")
    logger.info(f"  Adapter output: {output_dir}")
    logger.info(f"  Full model output: {output_dir}_full")
    logger.info("="*70)


if __name__ == "__main__":
    main()
