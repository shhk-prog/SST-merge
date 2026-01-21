# SST-Merge 補間モード (Interpolation Mode)

## 概要

Task Arithmetic互換の補間型マージを使用するSST-Mergeの実装です。

### マージ方式の違い

| モード | 式 | 特徴 |
|--------|-----|------|
| **加算型（従来）** | `merged = utility + α × safety` | Utilityが常に保持される |
| **補間型（新規）** | `merged = (1-α) × utility + α × safety` | Task Arithmeticと同じ動作 |

## ファイル構成

- `sst_merge_interpolation.py` - 補間型SST-Mergeクラス
- `run_all_merges_interpolation.py` - マージ実行スクリプト
- `run_all_evals_interpolation.py` - 評価実行スクリプト

## 出力ディレクトリ

```
merge_adapters_interpolation/    # アダプター出力
merge_model_interpolation/        # フルモデル出力
merge_eval_interpolation/         # 評価結果出力
```

## 使用方法

### 1. マージ実行

```bash
python3 run_all_merges_interpolation.py
```

出力例: `A6_A7_sst_interp_k5_nogevp_a1.0`

### 2. 評価実行

```bash
python3 run_all_evals_interpolation.py
```

### 3. 結果確認

```bash
# Jailbreak評価結果
cat merge_eval_interpolation/A6_A7_sst_interp_k5_nogevp_a1.0_jailbreak_eval_results.json

# Alpaca評価結果
cat merge_eval_interpolation/A6_A7_sst_interp_k5_nogevp_a1.0_alpaca_eval_results.json
```

## 期待される効果

従来の加算型SST-Merge（α=1.0）と比較して：

| 設定 | ASR（従来） | ASR（期待） | 改善 |
|------|------------|------------|------|
| SST-Merge α=1.0 | 18.6% | **~10.6%** | **-8.0%pt** |

理論的には、補間型α=1.0はTask Arithmetic α=0.5と同等の性能になるはずです。

## パラメータ設定

`run_all_merges_interpolation.py`で設定可能：

```python
alpha_values = [1.0]              # α値
sst_k_values = [5]                # FIM次元数
use_layerwise_options = [False]   # Layer-wise重み
use_gevp_options = [False]        # GEVPマスク
```

## 技術的詳細

### 補間型マージの実装

`sst_merge_interpolation.py`の`_interpolation_merge`メソッド：

```python
def _interpolation_merge(self, utility_adapter, safety_adapter):
    merged = {}
    alpha = self.safety_weight
    
    for key in utility_adapter.keys():
        if key in safety_adapter:
            utility_weight = 1.0 - alpha
            safety_weight = alpha
            merged[key] = utility_weight * utility_val + safety_weight * safety_val
        else:
            merged[key] = utility_val
    
    return merged
```

### GEVP併用時の動作

`use_gevp=True`の場合、マスクを使った補間型マージを実行：

```python
safety_weight = alpha * layer_weight * mask[i]
utility_weight = 1.0 - safety_weight
merged[i] = utility_weight * utility[i] + safety_weight * safety[i]
```

## 比較テーブル

| α設定 | Task Arithmetic | SST-Merge（加算） | SST-Merge（補間） |
|-------|----------------|------------------|------------------|
| 0.0 | 100% Utility | 100% Utility | 100% Utility |
| 0.5 | 50% U + 50% S | U + 50% S | **50% U + 50% S** |
| 1.0 | 100% Safety | U + 100% S | **100% Safety** |

✅ = 補間型はTask Arithmeticと同じ動作

## 注意事項

- 従来の加算型（`sst_merge.py`）とは**別の実装**です
- 出力ディレクトリも分離されているため、混在しません
- 既存の評価結果には影響しません

## 関連ドキュメント

- [性能差の分析レポート](file:///mnt/iag-02/home/hiromi/.gemini/antigravity/brain/1647e986-c504-460e-a47c-7d0d62c74a6e/performance_gap_analysis.md)
- [GEVP無効化の分析](file:///mnt/iag-02/home/hiromi/.gemini/antigravity/brain/1647e986-c504-460e-a47c-7d0d62c74a6e/analysis_gevp_disabled.md)
