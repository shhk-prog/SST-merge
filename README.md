# SST-Merge: Safety Subspace Task-Merge

安全サブスペース選択型Task Arithmetic (SST-Merge) の実装と実験コード。

## 📋 目次

1. [概要](#概要)
2. [開発フェーズ](#開発フェーズ)
3. [理論的背景](#理論的背景)
4. [プロジェクト構造](#プロジェクト構造)
5. [インストール](#インストール)
6. [クイックスタート](#クイックスタート)
7. [実験パターン](#実験パターン)
8. [コアアルゴリズム](#コアアルゴリズム)
9. [データセット](#データセット)
10. [評価指標](#評価指標)
11. [設定ファイル](#設定ファイル)
12. [ベースライン手法](#ベースライン手法)
13. [実験結果](#実験結果)
14. [サポートモデル](#サポートモデル)
15. [トラブルシューティング](#トラブルシューティング)
16. [参考文献](#参考文献)

---

## 概要

SST-Mergeは、**Fisher Information Matrix (FIM)** と **一般化固有値問題 (GEVP)** に基づく、LLMのLoRAアダプターマージング手法です。

### 解決する課題

従来のLoRAマージング手法（Task Arithmetic、TIES等）では、**Safety（安全性）とUtility（有用性）がトレードオフ**関係にあります。これを「**Safety Tax（安全税）**」と呼びます。

**例**: 安全性を10%向上させると、タスク性能が10%低下する

### SST-Mergeの解決策

**Utility（有用性）を固定し、Safetyを直交サブスペースに射影**することで、Safety Taxを最小化します。

| 項目 | 説明 |
|------|------|
| **目標** | Safety Tax 60-70%削減 |
| **手法** | GEVPによる安全サブスペース特定 |
| **特徴** | Utility固定 + Safety射影 |

---

## 開発フェーズ

本プロジェクトは、**研究フェーズ（Phase 1-3）** と **実装フェーズ（Phase 1-10）** の2つの体系で管理されています。

---

### 研究フェーズ（理論・検証）

研究の進行を管理するフェーズです。詳細は `guide/` フォルダを参照してください。

| フェーズ | 名称 | 目的 | 主要成果物 |
|----------|------|------|------------|
| **Phase 1** | 理論的検証と定式化の厳密化 | GEVPの数学的厳密性を証明 | 定式化シート、新規性証明 |
| **Phase 2** | 計算効率とスケーラビリティの検証 | LLMスケールでの実行可能性を確認 | FIM近似戦略、計算量分析 |
| **Phase 3** | 網羅的な実証実験とSOTA性能の確定 | ベースラインとの比較実験 | 実験結果、論文用データ |

#### Phase 1: 理論的検証

**目標**: SST-MergeのGEVP定式化が数学的に厳密であり、既存のSOTA手法（AlignGuard-LoRA, DARE等）と比較して理論的に優位であることを証明

**主要タスク**:
1. **Task 1.1**: GEVPの構成要素（F_harm, F_benign）の厳密な定義
   - Fisher Information Matrix (FIM) としての定義
   - 対称性、半正定値性の確認
   - 数値解法の安定性検証

2. **Task 1.2**: 理論的優位性（新規性）の証明
   - AlignGuard-LoRAとの差別化（単一FIM vs 二元最適化）
   - DAREとの差別化（幾何学 vs 統計的頑健性）

**成果物**:
- 理論的定式化シート
- 数値解法要件定義
- 新規性の厳密証明書

#### Phase 2: 計算効率の検証

**目標**: LLMスケールでGEVPを実行可能にするためのFIM近似戦略を検証

**主要タスク**:
1. **FIM近似戦略の検証**
   - LoRA勾配分散近似（O(N²) → O(N)への削減）
   - K-FAC低ランク近似
   - VILA原理によるパラメータ識別

2. **スケーラビリティテスト**
   - 7B, 8B, 14Bモデルでの実行時間測定
   - メモリ使用量の分析

**成果物**:
- 計算効率レポート
- 推奨近似手法の選定

#### Phase 3: 実証実験

**目標**: 3つの実験でSOTA性能を確定

**実験内容**:
1. **実験1**: Safety Tax定量化（Safety-Utilityトレードオフ）
2. **実験2**: マルチタスク干渉耐性（8-20エキスパート）
3. **実験3**: ベースライン比較（TA, TIES, DARE, AlignGuard-LoRA）

**成果物**:
- 論文用の実験結果
- 可視化（パレート曲線、比較表）

---

### 実装フェーズ（コード開発）

ソフトウェア実装を管理するフェーズです。`scripts/test_end_to_end.py` で統合テストが実行されます。

| フェーズ | 名称 | 対応モジュール | 状態 |
|----------|------|----------------|------|
| **Phase 1-3** | LoRA基礎 | `model_loader.py`, `adapter_utils.py` | ✅ 完了 |
| **Phase 4-5** | FIM計算・GEVP解法 | `fim_calculator.py`, `gevp_solver.py` | ✅ 完了 |
| **Phase 6-7** | LoRA統合・マージ | `sst_merge.py`, `lora_trainer.py` | ✅ 完了 |
| **Phase 8** | 評価パイプライン | `evaluation/` | ✅ 完了 |
| **Phase 9** | ベンチマーク | `experiments/`, `baselines/` | ✅ 完了 |
| **Phase 10** | エンドツーエンド統合 | 全モジュール統合 | ✅ 完了 |

#### Phase 1-3: LoRA基礎

**実装内容**:
- モデルのダウンロード・ロード（Mistral-7B, Llama-3.1-8B, Qwen2.5-14B）
- LoRAアダプターのパラメータ抽出
- アダプターの保存・読み込み

**対応ファイル**:
- `src/utils/model_loader.py` - モデルローダー
- `src/adapter_utils.py` - アダプター管理
- `src/lora_trainer.py` - LoRAトレーニング

**テスト**:
```bash
python3 scripts/test_lora_basics.py
```

#### Phase 4-5: FIM計算・GEVP解法

**実装内容**:
- Fisher Information Matrix (FIM) の計算
  - 勾配分散近似（推奨）
  - K-FAC近似
  - VILA近似
- 一般化固有値問題（GEVP）の解法
  - scipy.linalg.eigh 使用
  - 安全サブスペースの選択

**対応ファイル**:
- `src/fim_calculator.py` - FIM計算
- `src/gevp_solver.py` - GEVPソルバー

**テスト**:
```bash
python3 scripts/test_fim_gevp.py
```

#### Phase 6-7: LoRA統合・マージ

**実装内容**:
- SST-Mergeアルゴリズムのコア実装
  - `merge_lora_adapters()` - 有害/良性マージ
  - `merge_utility_safety()` - Utility固定、Safety射影
- 安全サブスペースへの射影
- アダプターの結合

**対応ファイル**:
- `src/sst_merge.py` - SST-Mergeコア

**テスト**:
```bash
python3 scripts/test_sst_merge.py
```

#### Phase 8: 評価パイプライン

**実装内容**:
- 安全性評価（拒否率、Jailbreak耐性）
- ユーティリティ評価（MMLU精度、HumanEval Pass@1）
- Safety Tax計算
- メトリクスレポート生成

**対応ファイル**:
- `src/evaluation/safety_evaluator.py`
- `src/evaluation/utility_evaluator.py`
- `src/evaluation/safety_tax_calculator.py`
- `src/evaluation/metrics_reporter.py`

**テスト**:
```bash
python3 scripts/test_evaluation.py
```

#### Phase 9: ベンチマーク

**実装内容**:
- ベースライン手法の実装
  - Task Arithmetic, TIES-Merging, DARE, AlignGuard-LoRA
- 複数手法の比較
- パレート効率分析

**対応ファイル**:
- `src/baselines/` - ベースライン実装
- `src/baseline_methods.py` - 統合インターフェース
- `experiments/exp3_baseline_comparison.py`

#### Phase 10: エンドツーエンド統合

**実装内容**:
- Phase 1-9の統合テスト
- 完全なパイプラインの動作確認
- ダミーデータ・実データの両方での検証

**対応ファイル**:
- `scripts/test_end_to_end.py` - 統合テスト

**テスト**:
```bash
# 全フェーズの統合テスト
python3 scripts/test_end_to_end.py
```

---

### フェーズ対応表

研究フェーズと実装フェーズの対応関係:

```
研究フェーズ              実装フェーズ
─────────────────────────────────────────────────
Phase 1 (理論検証)    →   Phase 4-5 (FIM/GEVP)
                          理論を実装に落とし込み

Phase 2 (計算効率)    →   Phase 4-5 (FIM近似)
                          近似アルゴリズムの実装

Phase 3 (実証実験)    →   Phase 6-10 (マージ・評価・統合)
                          実験の実行と結果収集
```

---

## 理論的背景

### アルゴリズムの核心

SST-Mergeは以下の理論に基づいています：

```
1. F_utility: Utilityタスクで重要なパラメータ空間（FIM）
2. F_safety: Safetyタスクで重要なパラメータ空間（FIM）
3. GEVP: F_safety v = λ F_utility v を解く
4. 高固有値λ: Safety重要 かつ Utility非重要 → 安全に追加可能
5. マージ: Utility (固定) + α × Safety (射影)
```

### 数式による説明

**一般化固有値問題 (GEVP)**:

```
F_safety v = λ F_utility v
```

- `λ`が大きい固有ベクトル: Safetyにとって重要、Utilityにとって非重要
- これらの方向にのみSafetyパラメータを射影することで、Utility性能を維持

**射影演算**:

```
φ_projected = V_k V_k^T φ_safety
```

- `V_k`: 上位k個の固有ベクトル（安全サブスペースの基底）
- `φ_safety`: Safetyアダプターのパラメータ

**最終マージ**:

```
φ_merged = φ_utility + α × φ_projected
```

- `φ_utility`: Utilityアダプター（固定）
- `α`: Safety重み（0.0-1.0）

---

## プロジェクト構造

```
SST_merge/
├── src/                              # ソースコード
│   ├── sst_merge.py                  # SST-Mergeコア実装
│   ├── fim_calculator.py             # Fisher Information Matrix計算
│   ├── gevp_solver.py                # 一般化固有値問題ソルバー
│   ├── lora_trainer.py               # LoRAトレーニング
│   ├── adapter_utils.py              # アダプター保存・読み込み
│   ├── model_utils.py                # モデルユーティリティ
│   ├── baselines/                    # ベースライン手法
│   │   ├── task_arithmetic.py        # Task Arithmetic
│   │   ├── ties_merging.py           # TIES Merging
│   │   ├── dare.py                   # DARE
│   │   └── alignguard_lora.py        # AlignGuard-LoRA
│   ├── evaluation/                   # 評価モジュール
│   │   ├── safety_evaluator.py       # 安全性評価
│   │   ├── utility_evaluator.py      # 有用性評価
│   │   ├── safety_tax_calculator.py  # Safety Tax計算
│   │   └── metrics_reporter.py       # メトリクスレポート
│   └── utils/                        # ユーティリティ
│       ├── model_loader.py           # モデルローダー
│       ├── data_loader.py            # データローダー（BeaverTails, MMLU等）
│       ├── instruction_loaders.py    # 指示データローダー（RepliQA, Alpaca等）
│       └── task_specific_loaders.py  # タスク固有ローダー
│
├── experiments/                      # 実験スクリプト
│   ├── create_instruction_model.py   # A5/A6/A7アダプター作成
│   ├── run_sst_merge.py              # SST-Merge実行
│   ├── evaluate_instruction_models.py # モデル評価
│   ├── exp1_safety_utility_tradeoff.py # 実験1: Safety-Utility トレードオフ
│   ├── exp2_multitask_interference.py  # 実験2: マルチタスク干渉
│   └── exp3_baseline_comparison.py     # 実験3: ベースライン比較
│
├── configs/                          # 設定ファイル
│   ├── experiment_config.yaml        # 実験設定
│   └── experiment_config_real.yaml   # 本番実験設定
│
├── data/                             # データセット
│   └── response_dataframe.csv        # Jailbreak評価データ（1,400サンプル）
│
├── saved_adapters/                   # 保存されたアダプター
│   └── {model_name}/
│       └── utility_model/
│           ├── utility_model_A5.pt   # RepliQAアダプター
│           ├── utility_model_A6.pt   # Alpacaアダプター
│           ├── utility_model_A7.pt   # Securityアダプター
│           └── sst_merged_*.pt       # マージ済みアダプター
│
├── scripts/                          # ユーティリティスクリプト
│   ├── test_sst_merge.py             # SST-Mergeテスト
│   ├── test_fim_gevp.py              # FIM/GEVPテスト
│   └── download_datasets.py          # データセットダウンロード
│
├── results/                          # 実験結果
├── logs/                             # ログファイル
├── requirements.txt                  # 依存パッケージ
└── README.md                         # このファイル
```

---

## インストール

### 前提条件

- Python 3.9+
- CUDA 11.8+ (GPU使用時)
- 16GB+ VRAM (推奨)

### 手順

```bash
# 1. リポジトリをクローン（既にある場合はスキップ）
cd /path/to/SST_merge

# 2. 仮想環境作成
python3 -m venv sst
source sst/bin/activate

# 3. 依存パッケージインストール
pip install -r requirements.txt

# 4. (オプション) Flash Attention 2のインストール（高速化）
pip install flash-attn --no-build-isolation
```

### 依存パッケージ

```
# コア
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
scipy>=1.10.0
numpy>=1.24.0

# データ
datasets>=2.12.0
evaluate>=0.4.0

# 可視化
matplotlib>=3.7.0
seaborn>=0.12.0

# 設定
pyyaml>=6.0
hydra-core>=1.3.0
```

---

## クイックスタート

### Step 1: Utilityアダプター作成 (A5, A6)

```bash
# A5: RepliQA（質問応答）
python3 experiments/create_instruction_model.py \
    --model llama-3.1-8b \
    --task repliqa \
    --mode full

# A6: Alpaca（指示応答）
python3 experiments/create_instruction_model.py \
    --model llama-3.1-8b \
    --task alpaca \
    --mode full
```

### Step 2: Safetyアダプター作成 (A7)

```bash
# A7: Security（Jailbreak防御）
python3 experiments/create_instruction_model.py \
    --model llama-3.1-8b \
    --task security \
    --mode full
```

### Step 3: SST-Mergeでマージ

```bash
# A5 + A7: RepliQA性能を維持しつつSafety向上
python3 experiments/run_sst_merge.py \
    --model llama-3.1-8b \
    --variant A5+A7 \
    --k 10 \
    --alpha 0.5

# A6 + A7: Alpaca性能を維持しつつSafety向上
python3 experiments/run_sst_merge.py \
    --model llama-3.1-8b \
    --variant A6+A7 \
    --k 10 \
    --alpha 0.5

# A5 + A6 + A7: 全Utility性能を維持しつつSafety向上
python3 experiments/run_sst_merge.py \
    --model llama-3.1-8b \
    --variant A5+A6+A7 \
    --k 10 \
    --alpha 0.5
```

### Step 4: 評価

```bash
python3 experiments/evaluate_instruction_models.py \
    --model llama-3.1-8b
```

---

## 実験パターン

本プロジェクトでは、**3つの実験パターン**を用意しています。目的に応じて選択してください。

### パターン比較表

| パターン | 用途 | データ | モデル | 実行時間 | GPU必要 |
|----------|------|--------|--------|----------|---------|
| **1. ダミーデータ** | アルゴリズム検証・デバッグ | ランダム生成 | DummyLoRAModel | ~1分 | 不要 |
| **2. 実データ** | パイプライン検証 | BeaverTails, MMLU等 | 実モデル | ~30分 | 推奨 |
| **3. フルLoRA** | 本番実験 | RepliQA, Alpaca, Security | 実モデル+LoRAトレーニング | ~数時間 | 必須 |

---

### パターン1: ダミーデータでの実験

**目的**: アルゴリズムの動作確認、デバッグ、単体テスト

**特徴**:
- ランダム生成されたダミーデータを使用
- 軽量なDummyLoRAModelを使用
- GPUなしでも実行可能
- 数分で完了

**実行方法**:

```bash
# エンドツーエンドテスト
python3 scripts/test_end_to_end.py

# フル実験（ダミーモード）
python3 experiments/run_full_experiments.py --model mistral-7b --experiment all
```

**使用ファイル**:
- `scripts/test_end_to_end.py` - Phase 1-10の統合テスト
- `experiments/run_full_experiments.py` - 3つの実験をダミーデータで実行
- `scripts/test_sst_merge.py` - SST-Mergeの単体テスト
- `scripts/test_fim_gevp.py` - FIM/GEVPの単体テスト

**コード例**:

```python
# ダミーモデルとデータの作成
class DummyLoRAModel(nn.Module):
    def __init__(self, hidden_size=128, lora_rank=16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(hidden_size, lora_rank))
        self.lora_B = nn.Parameter(torch.randn(lora_rank, hidden_size))

# ダミーデータローダー
def create_dummy_dataloader(num_batches=10, batch_size=4, seq_length=32):
    data = []
    for _ in range(num_batches):
        batch = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
            "labels": torch.randint(0, 100, (batch_size, seq_length))
        }
        data.append(batch)
    return data

# SST-Mergeテスト
model = DummyLoRAModel()
harm_data = create_dummy_dataloader()
benign_data = create_dummy_dataloader()

merger = SSTMerge(k=10, device="cpu")
merged = merger.merge_lora_adapters(model, lora_adapters, harm_data, benign_data)
```

---

### パターン2: 実データでの実験

**目的**: 実際のデータセットを使用したパイプライン検証

**特徴**:
- BeaverTails、MMLU、HumanEval等の実データを使用
- 実際のLLMモデル（Mistral-7B、Llama-3.1-8B等）を使用
- `minimal`モードと`full`モードを選択可能
- 保存済みアダプターの再利用が可能

**実行方法**:

```bash
# minimalモード（デバッグ用、少量データ）
python3 experiments/run_real_experiments.py \
    --mode minimal \
    --model mistral-7b \
    --experiment exp1

# fullモード（本番用、全データ）
python3 experiments/run_real_experiments.py \
    --mode full \
    --model llama-3.1-8b \
    --experiment all

# 保存済みアダプターを使用（トレーニングをスキップ）
python3 experiments/run_real_experiments.py \
    --mode full \
    --model llama-3.1-8b \
    --use-saved-adapters
```

**使用ファイル**:
- `experiments/run_real_experiments.py` - 実データ実験メインスクリプト
- `configs/experiment_config_real.yaml` - 実験設定ファイル

**モード比較**:

| 項目 | minimal | full |
|------|---------|------|
| BeaverTails | 100サンプル | 10,000サンプル |
| MMLU | 50サンプル | 1,000サンプル |
| HumanEval | 10サンプル | 164サンプル |
| 評価バッチ数 | 10 | 全バッチ |
| 用途 | デバッグ | 本番実験 |

**コード例**:

```python
from experiments.run_real_experiments import RealDataExperiment

# 実験インスタンス作成
experiment = RealDataExperiment(
    config_path="configs/experiment_config_real.yaml",
    mode="minimal",           # or "full"
    model_name="llama-3.1-8b",
    use_saved_adapters=True   # 保存済みアダプターを使用
)

# データセット読み込み
datasets = experiment.load_datasets()
# → beavertails_train, beavertails_eval, mmlu, humaneval

# モデル読み込み
model, tokenizer, loader = experiment.load_model()

# 実験1: Safety Tax定量化
experiment.run_experiment_1(datasets, model, tokenizer)

# 実験2: マルチタスク干渉
experiment.run_experiment_2(datasets, model, tokenizer)

# 実験3: ベースライン比較
experiment.run_experiment_3(datasets, model, tokenizer)
```

---

### パターン3: 実際のLoRAトレーニングでの実験

**目的**: LoRAアダプターをゼロから作成し、SST-Mergeを実行

**特徴**:
- RepliQA (A5)、Alpaca (A6)、Security (A7)データでLoRAトレーニング
- Unslothベストプラクティスに準拠したトレーニング設定
- 完全な実験パイプライン
- アダプターの保存・再利用が可能

**実行方法**:

```bash
# Step 1: Utilityアダプター作成
python3 experiments/create_instruction_model.py \
    --model llama-3.1-8b \
    --task repliqa \    # A5
    --mode full

python3 experiments/create_instruction_model.py \
    --model llama-3.1-8b \
    --task alpaca \     # A6
    --mode full

# Step 2: Safetyアダプター作成
python3 experiments/create_instruction_model.py \
    --model llama-3.1-8b \
    --task security \   # A7
    --mode full

# Step 3: SST-Mergeでマージ
python3 experiments/run_sst_merge.py \
    --model llama-3.1-8b \
    --variant A5+A6+A7 \
    --k 10 \
    --alpha 0.5

# Step 4: 評価
python3 experiments/evaluate_instruction_models.py \
    --model llama-3.1-8b
```

**使用ファイル**:
- `experiments/create_instruction_model.py` - LoRAアダプター作成
- `experiments/run_sst_merge.py` - SST-Mergeマージ実行
- `experiments/evaluate_instruction_models.py` - モデル評価
- `src/lora_trainer.py` - LoRAトレーニングロジック

**アダプター保存場所**:

```
saved_adapters/
└── llama-3.1-8b/
    └── utility_model/
        ├── utility_model_A5.pt     # RepliQA
        ├── utility_model_A6.pt     # Alpaca
        ├── utility_model_A7.pt     # Security
        └── sst_merged_A5_A6_A7_k10_alpha0.50.pt  # マージ済み
```

**コード例**:

```python
from src.utils.model_loader import ModelLoader
from src.lora_trainer import LoRATrainer
from src.utils.instruction_loaders import load_repliqa, load_alpaca, load_security
from src.sst_merge import SSTMerge
from src.adapter_utils import load_lora_adapter, save_lora_adapter

# モデル読み込み
loader = ModelLoader("llama-3.1-8b")
model, tokenizer = loader.load_model()

# LoRAトレーナー
trainer = LoRATrainer(model, tokenizer, device='cuda')

# A5: RepliQAでトレーニング
repliqa_data = load_repliqa(split='train', batch_size=32)
A5_adapter = trainer.train_lora_adapter(
    dataloader=repliqa_data,
    task_type='benign',
    num_epochs=3,
    lora_r=32,
    lora_alpha=64
)

# A6: Alpacaでトレーニング
alpaca_data = load_alpaca(split='train', batch_size=32)
A6_adapter = trainer.train_lora_adapter(
    dataloader=alpaca_data,
    task_type='benign',
    num_epochs=3
)

# A7: Securityでトレーニング
security_data = load_security(csv_path='data/response_dataframe.csv')
A7_adapter = trainer.train_lora_adapter(
    dataloader=security_data,
    task_type='safety',
    num_epochs=3
)

# SST-Merge
sst_merge = SSTMerge(k=10, device='cuda')
merged_adapter = sst_merge.merge_utility_safety(
    model=model,
    utility_adapters=[A5_adapter, A6_adapter],
    safety_adapter=A7_adapter,
    utility_dataloader=combine_dataloaders([repliqa_data, alpaca_data]),
    safety_dataloader=security_data,
    alpha=0.5
)

# 保存
save_lora_adapter(merged_adapter, 'saved_adapters/llama-3.1-8b/utility_model/sst_merged.pt')
```

---

### パターン選択ガイド

```
開発フェーズに応じた選択:

┌─────────────────┐
│  新機能開発     │ → パターン1: ダミーデータ
│  デバッグ       │   （高速イテレーション）
└────────┬────────┘
         ↓
┌─────────────────┐
│  パイプライン   │ → パターン2: 実データ (minimal)
│  動作確認       │   （実データで検証）
└────────┬────────┘
         ↓
┌─────────────────┐
│  本番実験       │ → パターン2: 実データ (full)
│  論文用データ   │   または
└────────┬────────┘   パターン3: フルLoRA
         ↓
┌─────────────────┐
│  最終評価       │ → パターン3: フルLoRA
│  モデル公開     │   （完全な再現性）
└─────────────────┘
```

---

## コアアルゴリズム

### SSTMerge クラス

```python
from src.sst_merge import SSTMerge
from src.adapter_utils import load_lora_adapter

# アダプターロード
A5_adapter, _ = load_lora_adapter('saved_adapters/llama-3.1-8b/utility_model/utility_model_A5.pt')
A6_adapter, _ = load_lora_adapter('saved_adapters/llama-3.1-8b/utility_model/utility_model_A6.pt')
A7_adapter, _ = load_lora_adapter('saved_adapters/llama-3.1-8b/utility_model/utility_model_A7.pt')

# SST-Merge初期化
sst_merge = SSTMerge(
    k=10,                              # 安全サブスペースの次元数
    fim_approximation="gradient_variance",  # FIM近似手法
    regularization=1e-6,               # 正則化項
    device="cuda"
)

# マージ実行
merged_adapter = sst_merge.merge_utility_safety(
    model=base_model,
    utility_adapters=[A5_adapter, A6_adapter],  # 固定
    safety_adapter=A7_adapter,                   # 射影
    utility_dataloader=utility_dl,
    safety_dataloader=safety_dl,
    alpha=0.5                                    # Safety重み
)
```

### FIMCalculator クラス

Fisher Information Matrixの計算を担当：

```python
from src.fim_calculator import FIMCalculator

fim_calculator = FIMCalculator(
    model=peft_model,
    approximation="gradient_variance",  # "gradient_variance", "kfac", "vila"
    regularization=1e-6,
    device="cuda"
)

# FIM計算
F_utility = fim_calculator.compute_fim_benign(utility_dataloader, max_samples=1000)
F_safety = fim_calculator.compute_fim_harm(safety_dataloader, max_samples=1000)
```

**FIM近似手法**:

| 手法 | 説明 | 計算量 |
|------|------|--------|
| `gradient_variance` | LoRA勾配分散近似（推奨） | O(N) |
| `kfac` | K-FAC近似 | O(N²) |
| `vila` | VILA原理によるパラメータ選択 | O(N) |

### GEVPSolver クラス

一般化固有値問題を解いて安全サブスペースを特定：

```python
from src.gevp_solver import GEVPSolver

gevp_solver = GEVPSolver(
    regularization=1e-6,
    use_scipy=True  # scipyを使用（より安定）
)

# GEVP解く: F_safety v = λ F_utility v
eigenvalues, eigenvectors = gevp_solver.solve_gevp(
    F_safety, 
    F_utility, 
    k=10  # 上位k個の固有値・固有ベクトル
)

# 安全サブスペースを選択
safety_subspace = gevp_solver.select_safety_subspace(eigenvectors, k=10)
```

### LoRATrainer クラス

LoRAアダプターのトレーニング（Unslothベストプラクティス準拠）：

```python
from src.lora_trainer import LoRATrainer

trainer = LoRATrainer(
    base_model=model,
    tokenizer=tokenizer,
    device='cuda'
)

# トレーニング
adapter = trainer.train_lora_adapter(
    dataloader=task_data,
    task_type='benign',       # 'harmful', 'benign', 'safety'
    num_epochs=3,
    learning_rate=2e-4,
    lora_r=32,                # LoRA rank（Unsloth推奨: 16 or 32）
    lora_alpha=64,            # 2 × r
    lora_dropout=0.0,         # Unsloth推奨: 0
    weight_decay=0.01,
    warmup_ratio=0.1,         # 5-10% of steps
    gradient_accumulation_steps=4
)

# 保存
trainer.save_adapter(adapter, 'path/to/adapter.pt')
```

**LoRA設定のポイント（Unslothベストプラクティス）**:

```python
lora_config = LoraConfig(
    r=32,              # Rank（16 or 32推奨）
    lora_alpha=64,     # 2 × r
    lora_dropout=0.0,  # デフォルト0
    target_modules=[   # 全主要レイヤーをターゲット
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# Train on completions only（プロンプト部分をマスク）
labels = inputs['input_ids'].clone()
labels[:, :prompt_len] = -100  # プロンプト部分を損失計算から除外
```

---

## データセット

### アダプター作成用データセット

| ID | 名前 | 用途 | データソース | サンプル数 |
|----|------|------|--------------|------------|
| A5 | RepliQA | 質問応答（Utility） | ServiceNow/repliqa | ~10,000 |
| A6 | Alpaca | 指示応答（Utility） | tatsu-lab/alpaca | ~52,000 |
| A7 | Security | Jailbreak防御（Safety） | data/response_dataframe.csv | 1,400 |

### 評価用データセット

| 名前 | 用途 | データソース |
|------|------|--------------|
| BeaverTails | 安全性評価 | PKU-Alignment/BeaverTails |
| MMLU | 一般知識評価 | cais/mmlu |
| HumanEval | コーディング評価 | openai_humaneval |

### データローダー使用例

```python
from src.utils.data_loader import load_beavertails, load_mmlu
from src.utils.instruction_loaders import load_repliqa, load_alpaca, load_security

# BeaverTails（安全性評価）
safety_loader = load_beavertails(split='train', max_samples=1000, batch_size=32)

# MMLU（一般知識）
mmlu_loader = load_mmlu(subjects='all', split='test', max_samples=1000, batch_size=32)

# RepliQA（質問応答）
repliqa_loader = load_repliqa(split='train', max_samples=1000, batch_size=32)

# Alpaca（指示応答）
alpaca_loader = load_alpaca(split='train', max_samples=1000, batch_size=32)

# Security（Jailbreak防御）
security_loader = load_security(csv_path='data/response_dataframe.csv', batch_size=32)
```

---

## 評価指標

### Safety Tax Calculator

Safety Taxとアライメントドリフトを定量化：

```python
from src.evaluation.safety_tax_calculator import SafetyTaxCalculator

calculator = SafetyTaxCalculator(
    baseline_method="AlignGuard-LoRA",
    target_reduction=0.65  # 目標: 65%削減
)

metrics = calculator.compute_safety_tax(
    safety_before=0.70,   # マージ前の安全性
    safety_after=0.90,    # マージ後の安全性
    utility_before=0.95,  # マージ前のUtility
    utility_after=0.93,   # マージ後のUtility
    method_name="SST-Merge"
)

print(f"Safety Tax: {metrics.safety_tax:.4f}")
print(f"Utility Drop Rate: {metrics.utility_drop_rate:.2%}")
print(f"Safety Gain Rate: {metrics.safety_gain_rate:.2%}")
print(f"Alignment Drift: {metrics.alignment_drift:.4f}")
```

### 評価指標一覧

| 指標 | 説明 | 目標 |
|------|------|------|
| **Safety Tax** | (Utility低下率) / (Safety向上率) | 低いほど良い |
| **Alignment Drift** | \|Safety_after - Safety_before\| / Safety_before | 小さいほど安定 |
| **Utility Drop Rate** | (Utility_before - Utility_after) / Utility_before | 小さいほど良い |
| **Refusal Rate** | Jailbreak攻撃への拒否率 | 高いほど良い |
| **MMLU Accuracy** | 一般知識の正答率 | 高いほど良い |

### パレート効率分析

複数手法のSafety-Utilityトレードオフを可視化：

```python
pareto = calculator.compute_pareto_efficiency(
    safety_scores=[0.90, 0.85, 0.80],
    utility_scores=[0.93, 0.82, 0.85],
    method_names=["SST-Merge", "AlignGuard-LoRA", "DARE"]
)

print(f"Best method: {pareto['best_method']}")
print(f"Pareto front: {pareto['pareto_front']}")
```

---

## 設定ファイル

### experiment_config.yaml

```yaml
# モデル設定
model:
  base_model: "meta-llama/Llama-3.1-8B-Instruct"
  device: "cuda"
  dtype: "float16"

# データセット設定
datasets:
  safety:
    name: "PKU-Alignment/BeaverTails"
    max_samples: 10000
  utility:
    mmlu:
      name: "cais/mmlu"
      subjects: ["abstract_algebra", "anatomy", "astronomy"]

# LoRA設定
lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.0
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]

# SST-Merge設定
sst_merge:
  k: 10                              # 安全サブスペース次元数
  fim_approximation: "gradient_variance"
  regularization: 1e-6

# 評価設定
evaluation:
  batch_size: 8
  max_length: 512
  metrics:
    - "refusal_rate"
    - "mmlu_accuracy"
    - "safety_tax"
```

---

## ベースライン手法

SST-Mergeと比較するベースライン手法は、**2つの実装アプローチ**で提供されています。

### 実装アプローチ比較

| アプローチ | 説明 | ファイル | 利点 |
|------------|------|----------|------|
| **mergekit/PEFT** | HuggingFace PEFTのadd_weighted_adapter使用 | `src/mergekit_wrapper.py` | 標準的、互換性高い |
| **カスタム実装** | 論文に基づく独自実装 | `src/baselines/`, `src/baseline_methods.py` | 細かい制御、研究用途 |

---

### アプローチ1: mergekit/PEFT 使用

HuggingFace PEFTライブラリの`add_weighted_adapter`機能を使用したマージ。
標準的で互換性が高く、実運用に適しています。

**使用ファイル**: `src/mergekit_wrapper.py`

```python
from src.mergekit_wrapper import MergekitWrapper

wrapper = MergekitWrapper()

# PEFTでマージ（combination_typeで手法を選択）
merged_model = wrapper.merge_with_peft(
    base_model=model,
    adapters=adapters,
    adapter_names=["adapter1", "adapter2", "adapter3"],
    weights=[0.33, 0.33, 0.34],
    combination_type="linear"  # "linear", "ties", "dare_linear", "dare_ties"
)
```

**サポートされるcombination_type**:

| タイプ | 説明 |
|--------|------|
| `linear` | Task Arithmetic（単純な重み付き平均） |
| `ties` | TIES-Merging（符号競合解決） |
| `dare_linear` | DARE + 線形マージ |
| `dare_ties` | DARE + TIES |

**インストール確認**:

```python
from src.mergekit_wrapper import check_mergekit_installation

info = check_mergekit_installation()
# {'mergekit': True/False, 'peft': True/False, 'message': [...]}
```

---

### アプローチ2: カスタム実装

論文に基づく独自実装。アルゴリズムの詳細な制御が可能で、研究用途に適しています。

**使用ファイル**: 
- `src/baselines/` - 詳細なベースライン実装
- `src/baseline_methods.py` - 統合インターフェース

---

#### Task Arithmetic (TA)

**論文**: [Editing Models with Task Arithmetic (ICLR 2023)](https://arxiv.org/abs/2212.04089)

**特徴**: 最もシンプル、タスクベクトルの線形結合

**数式**: `θ_merged = θ_base + Σ λ_i (θ_i - θ_base)`

```python
# カスタム実装 (src/baselines/task_arithmetic.py)
from src.baselines.task_arithmetic import TaskArithmetic

ta = TaskArithmetic(scaling_factor=0.5)
merged = ta.merge(lora_adapters=[A5, A6, A7])

# または統合インターフェース (src/baseline_methods.py)
from src.baseline_methods import TaskArithmetic as TA

ta = TA()
merged = ta.merge(adapters=[A5, A6, A7], weights=[0.33, 0.33, 0.34])
```

---

#### TIES Merging

**論文**: [TIES-Merging: Resolving Interference When Merging Models (NeurIPS 2023)](https://arxiv.org/abs/2306.01708)

**特徴**: 符号の競合を解決、小さい値をトリミング

**アルゴリズム**:
1. **Trim**: 小さい更新をゼロにする
2. **Elect Sign**: 符号の多数決
3. **Merge**: 一致した符号のパラメータを結合

```python
# カスタム実装 (src/baselines/ties_merging.py)
from src.baselines.ties_merging import TIESMerging

ties = TIESMerging(trim_threshold=0.2)  # 下位20%をトリミング
merged = ties.merge(lora_adapters=[A5, A6, A7])

# または統合インターフェース
from src.baseline_methods import TIESMerging

ties = TIESMerging(density=0.2)  # 上位20%を保持
merged = ties.merge(adapters=[A5, A6, A7])
```

---

#### DARE (Drop And REscale)

**論文**: [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)

**特徴**: SVDベースのサブスペース抽出、ドロップアウトとリスケーリング

**アルゴリズム**:
1. タスクベクトルをSVD分解
2. ランダムにパラメータをドロップ
3. リスケーリングで期待値を維持
4. 重み付きマージ

```python
# カスタム実装 (src/baselines/dare.py)
from src.baselines.dare import DARE

dare = DARE(
    k=10,           # サブスペース次元数
    drop_rate=0.5,  # ドロップ率
    rescale=True    # リスケーリング有効
)
merged = dare.merge_lora_adapters(base_params, [A5, A6, A7])

# Subspace Boosting（大規模エキスパート）
merged = dare.merge_with_subspace_boosting(
    base_params, 
    lora_adapters, 
    num_experts=20  # 20エキスパートで85%性能維持
)

# または統合インターフェース
from src.baseline_methods import DAREMerging

dare = DAREMerging(drop_rate=0.9)
merged = dare.merge(adapters=[A5, A6, A7])
```

---

#### AlignGuard-LoRA

**論文**: [AlignGuard-LoRA: Alignment-Preserving Fine-Tuning](https://huggingface.co/papers/2508.02079)

**特徴**: Fisher-Guided分解で有害方向を回避、50%のSafety Tax削減

**アルゴリズム**:
1. 有害データに対するFIM `F_harm` を計算
2. FIMの固有値分解 `F_harm = Q Λ Q^T`
3. 上位k個の固有ベクトル（有害方向）を特定
4. LoRAパラメータを有害方向から遠ざける
5. マージ

```python
from src.baselines.alignguard_lora import AlignGuardLoRA

agl = AlignGuardLoRA(
    top_k_harmful=5,        # 回避する有害方向の数
    avoidance_strength=0.8,  # 回避の強度
    regularization=1e-6
)

# マージ
merged = agl.merge_lora_adapters(
    base_model_params=base_params,
    lora_adapters=[A5, A6, A7],
    harm_dataloader=harm_data,
    max_samples=1000
)

# Safety Tax計算
metrics = agl.compute_safety_tax(
    original_safety=0.7,
    original_utility=0.9,
    merged_safety=0.85,
    merged_utility=0.85
)
# → {'safety_tax': 0.23, 'alignment_drift_reduction': 0.5}
```

---

### 手法比較表

| 手法 | Safety Tax | Utility維持 | 計算コスト | 特徴 |
|------|------------|-------------|------------|------|
| Task Arithmetic | 高い | 低い | 低い | 最もシンプル |
| TIES-Merging | 中程度 | 中程度 | 中程度 | 符号競合解決 |
| DARE | 中程度 | 高い | 中程度 | ドロップアウト正則化 |
| AlignGuard-LoRA | **50%削減** | 中程度 | 高い | FIMベース |
| **SST-Merge** | **60-70%削減** | **高い** | 高い | GEVP + Utility固定 |

---

### ベースライン実験の実行

```bash
# 全ベースライン比較
python3 experiments/exp3_baseline_comparison.py

# または
python3 experiments/run_baseline_experiments.py \
    --model llama-3.1-8b \
    --methods ta,ties,dare,agl,sst
```

---

## 実験結果

### 期待される結果

| モデル | Utility | Safety | Safety Tax | 削減率 |
|--------|---------|--------|-----------|--------|
| Base | 70% | 80% | - | - |
| Utility (A5+A6) | 95% | 80% | - | - |
| + Safety (Linear) | 85% | 95% | 10% | - |
| + Safety (SST) | **93%** | **95%** | **2%** | **80%** |

### 実験の実行

```bash
# 実験1: Safety-Utilityトレードオフ
python3 experiments/exp1_safety_utility_tradeoff.py

# 実験2: マルチタスク干渉
python3 experiments/exp2_multitask_interference.py

# 実験3: ベースライン比較
python3 experiments/exp3_baseline_comparison.py

# 全実験一括実行
bash run_all_experiments.sh
```

---

## サポートモデル

| モデル | エイリアス | パラメータ数 | 推奨VRAM |
|--------|-----------|------------|----------|
| Llama-3.1-8B-Instruct | `llama-3.1-8b` | 8B | 18GB |
| Mistral-7B-Instruct-v0.2 | `mistral-7b-v0.2` | 7B | 16GB |
| Qwen2.5-14B-Instruct | `qwen-2.5-14b` | 14B | 32GB |

### モデルローダー使用例

```python
from src.utils.model_loader import ModelLoader

loader = ModelLoader(
    model_name="llama-3.1-8b",  # エイリアス使用可能
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_flash_attention=True
)

model, tokenizer = loader.load_model()
```

---

## トラブルシューティング

### よくある問題

**1. CUDAメモリ不足**

```bash
# 解決策: バッチサイズを減らす、勾配チェックポイント有効化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**2. BeaverTailsデータセットのロードエラー**

```python
# BeaverTailsのsplit名: '30k_train', '30k_test'（'train', 'test'ではない）
# data_loader.pyで自動対応済み
```

**3. Flash Attention未インストール**

```bash
pip install flash-attn --no-build-isolation
```

**4. LoRAパラメータの勾配がNone**

```python
# モデルをtrainモードに設定
model.train()
# 勾配を有効化
for param in model.parameters():
    param.requires_grad = True
```

---

## ライセンス

MIT License

---

## 引用

```bibtex
@article{sst-merge-2025,
  title={Safety Subspace Task-Merge: GEVP-based LoRA Merging for Safety-Utility Trade-off Optimization},
  author={[Your Name]},
  year={2025}
}
```

---

## 参考文献

- [Unsloth LoRA Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Task Arithmetic (ICLR 2023)](https://arxiv.org/abs/2212.04089)
- [TIES-Merging](https://arxiv.org/abs/2306.01708)
- [DARE](https://arxiv.org/abs/2311.03099)
- [QLoRA Paper](https://arxiv.org/pdf/2305.14314)
- [rsLoRA Paper](https://arxiv.org/abs/2312.03732)
