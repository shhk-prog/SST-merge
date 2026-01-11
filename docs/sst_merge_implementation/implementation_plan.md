# SST-Merge実装計画：Phase 3 実証実験の実装

## 概要

本計画は、SST-Merge (Safety Subspace Task-Merge) の理論的基礎（Phase 1-2で確立）を実証するための包括的な実験実装を定義します。GEVPに基づく二元最適化アプローチが、既存手法（AlignGuard-LoRA、DARE、TIES-Merging）を超えるSafety Tax解消能力を持つことを実証します。

## ユーザーレビュー必須項目

> [!IMPORTANT]
> **データセットとモデルの選択**
> 
> 本実装では以下のリソースを使用します：
> - **ベースモデル**: Llama-2-7B または Mistral-7B（計算リソースに応じて選択）
> - **安全性データセット**: BeaverTails（有害/良性の分類済み）
> - **評価ベンチマーク**: 
>   - 安全性: BeaverTails評価セット
>   - ユーティリティ: MMLU（推論能力）、HumanEval（コーディング能力）
>   - Safety Tax: DRIFTCHECK（アライメントドリフト測定）
> 
> 計算リソースの制約により、フルスケール実験ではなくプロトタイプ実装を優先します。

> [!WARNING]
> **計算コストの考慮**
> 
> FIM計算は計算集約的です。本実装では以下の近似戦略を採用：
> 1. LoRA勾配分散近似（計算量をO(N²)→O(N)に削減）
> 2. VILA原理によるパラメータ識別（重要パラメータのみに焦点）
> 3. K-FAC低ランク近似（数値安定性の確保）

## 提案する変更

### コンポーネント1: Fisher Information Matrix計算

#### [NEW] [fim_calculator.py](file:///Users/saki/lab/src/SST_merge/src/fim_calculator.py)

Fisher Information Matrixの効率的な計算を実装します。

**主要機能**:
- `compute_fim_harm()`: 有害データセットに対するFIM計算
- `compute_fim_benign()`: 良性データセットに対するFIM計算
- `lora_gradient_variance_approximation()`: LoRA特化型の勾配分散近似
- `vila_parameter_identification()`: タスククリティカルなパラメータの識別

**技術的詳細**:
- LoRAの低ランク構造（ΔW = BA^T）を活用
- 勾配の共分散行列として経験的FIMを計算
- メモリ効率のためにブロック対角近似を使用

---

### コンポーネント2: GEVP（一般化固有値問題）ソルバー

#### [NEW] [gevp_solver.py](file:///Users/saki/lab/src/SST_merge/src/gevp_solver.py)

一般化固有値問題 F_harm v = λ F_benign v を解くモジュール。

**主要機能**:
- `solve_gevp()`: GEVPの数値解法（scipy.linalg.eighを使用）
- `select_safety_subspace()`: 上位k個の固有ベクトルで安全サブスペースを構築
- `compute_safety_efficiency()`: 各方向の安全効率λを計算

**技術的詳細**:
- F_benignの正定値性を確保（正則化項の追加）
- Generalized Krylov Subspace法による効率的な固有値計算
- 数値安定性のためのCholesky分解の活用

---

### コンポーネント3: SST-Mergeマージングアルゴリズム

#### [NEW] [sst_merge.py](file:///Users/saki/lab/src/SST_merge/src/sst_merge.py)

SST-Mergeのコアアルゴリズムを実装します。

**主要機能**:
- `merge_lora_adapters()`: 複数のLoRAアダプタをマージ
- `project_to_safety_subspace()`: LoRAパッチを安全サブスペースに射影
- `optimize_merge_coefficients()`: サブスペース内で最適な係数を決定

**アルゴリズムフロー**:
1. 各LoRAアダプタに対してF_harmとF_benignを計算
2. GEVPを解いて安全サブスペースS_safetyを特定
3. 元のLoRAパッチφ_originalをS_safetyに射影
4. 射影されたパッチφ_mergedを返す

---

### コンポーネント4: ベースライン実装

#### [NEW] [baselines/](file:///Users/saki/lab/src/SST_merge/src/baselines/)

比較のための既存手法の実装。

**実装する手法**:
- **Task Arithmetic (TA)**: 単純な線形結合
- **TIES-Merging**: Trim, Elect Sign, Merge
- **DARE**: SVDベースのSubspace Boosting
- **AlignGuard-LoRA (簡易版)**: 単一FIMの固有値分解による回避戦略

---

### コンポーネント5: 評価フレームワーク

#### [NEW] [evaluation/](file:///Users/saki/lab/src/SST_merge/src/evaluation/)

包括的な評価システム。

**評価モジュール**:
- `safety_evaluator.py`: BeaverTailsでの有害コンテンツ拒否率測定
- `utility_evaluator.py`: MMLU/HumanEvalでのタスク性能測定
- `safety_tax_calculator.py`: Safety Taxの定量化
- `metrics_reporter.py`: 複合メトリック（MedOmni-45°スタイル）の計算

**評価指標**:
- **安全性**: Refusal Rate（拒否率）、Jailbreak耐性
- **ユーティリティ**: MMLU精度、HumanEval Pass@1
- **Safety Tax**: アライメントドリフト率（DRIFTCHECK）
- **複合性能**: 安全性-ユーティリティのパレート効率

---

### コンポーネント6: 実験実行スクリプト

#### [NEW] [experiments/](file:///Users/saki/lab/src/SST_merge/experiments/)

再現可能な実験パイプライン。

**実験スクリプト**:
- `exp1_safety_utility_tradeoff.py`: Safety Tax定量化実験
- `exp2_multitask_interference.py`: マルチタスク干渉耐性実験
- `exp3_baseline_comparison.py`: ベースライン比較実験
- `run_all_experiments.sh`: 全実験の自動実行

---

## プロジェクト構造

```
SST_merge/
├── src/
│   ├── fim_calculator.py          # FIM計算モジュール
│   ├── gevp_solver.py             # GEVPソルバー
│   ├── sst_merge.py               # SST-Mergeコアアルゴリズム
│   ├── baselines/                 # ベースライン手法
│   │   ├── task_arithmetic.py
│   │   ├── ties_merging.py
│   │   ├── dare.py
│   │   └── alignguard_lora.py
│   ├── evaluation/                # 評価フレームワーク
│   │   ├── safety_evaluator.py
│   │   ├── utility_evaluator.py
│   │   ├── safety_tax_calculator.py
│   │   └── metrics_reporter.py
│   └── utils/                     # ユーティリティ
│       ├── lora_utils.py
│       └── data_loader.py
├── experiments/                   # 実験スクリプト
│   ├── exp1_safety_utility_tradeoff.py
│   ├── exp2_multitask_interference.py
│   ├── exp3_baseline_comparison.py
│   └── run_all_experiments.sh
├── configs/                       # 設定ファイル
│   └── experiment_config.yaml
├── data/                          # データセット（gitignore）
├── results/                       # 実験結果（gitignore）
├── docs/                          # ドキュメント
│   └── sst_merge_implementation/  # 本実装計画を保存
├── requirements.txt
└── README.md
```

## 検証計画

### 自動テスト

#### 1. 単体テスト

```bash
# FIM計算の正確性テスト
python -m pytest tests/test_fim_calculator.py -v

# GEVPソルバーの数値安定性テスト
python -m pytest tests/test_gevp_solver.py -v

# SST-Mergeアルゴリズムのテスト
python -m pytest tests/test_sst_merge.py -v
```

**テスト内容**:
- FIMが半正定値行列であることの検証
- GEVPの固有値が正であることの検証
- 安全サブスペースへの射影が直交性を保つことの検証

#### 2. 統合テスト

```bash
# エンドツーエンドのマージングパイプラインテスト
python -m pytest tests/test_integration.py -v
```

**テスト内容**:
- 小規模モデル（DistilBERT）での完全なマージングフロー
- ベースラインとの出力比較
- 計算時間の測定

### 実験的検証

#### 実験1: Safety Tax定量化

```bash
cd experiments
python exp1_safety_utility_tradeoff.py --config ../configs/experiment_config.yaml
```

**検証内容**:
- SST-MergeがAGLに対して50%以上のSafety Tax削減を達成するか
- 複合メトリック空間でパレート最適に近いか

**期待される結果**:
- AlignGuard-LoRA: アライメントドリフト50%削減
- SST-Merge: アライメントドリフト**60-70%削減**（理論的優位性の実証）

#### 実験2: マルチタスク干渉耐性

```bash
python exp2_multitask_interference.py --num_experts 8
```

**検証内容**:
- 8-20個のLoRAアダプタをマージした際の性能維持率
- DAREとの比較（統計的頑健性の検証）

**期待される結果**:
- DARE: 20エキスパートで平均性能85%維持
- SST-Merge: 20エキスパートで平均性能**88-90%維持**（FIMによる頑健性）

#### 実験3: ベースライン比較

```bash
python exp3_baseline_comparison.py --methods all
```

**検証内容**:
- TA, TIES, DARE, AGL, SST-Mergeの5手法を統一ベンチマークで比較
- 安全性、ユーティリティ、計算時間の3軸評価

**期待される結果**:
- SST-Mergeが安全性-ユーティリティのトレードオフで最良
- 計算時間はDAREと同等（O(N)の効率性を確認）

### 手動検証

実験完了後、以下を手動で確認：

1. **結果の可視化**: `results/`フォルダ内の図表を確認
   - Safety-Utility散布図でSST-Mergeがパレートフロンティアに位置するか
   - Safety Tax削減率の棒グラフ

2. **ログの確認**: 各実験のログファイルでエラーや警告がないか確認

3. **再現性**: 同じ設定で実験を再実行し、結果が一貫しているか確認

## 実装の優先順位

1. **Phase 1** (最優先): FIM計算とGEVPソルバーの基本実装
2. **Phase 2**: SST-Mergeコアアルゴリズムの実装
3. **Phase 3**: ベースライン実装と評価フレームワーク
4. **Phase 4**: 実験実行とデバッグ
5. **Phase 5**: 結果分析とドキュメント化

## 依存関係

```
torch>=2.0.0
transformers>=4.30.0
scipy>=1.10.0
numpy>=1.24.0
datasets>=2.12.0
peft>=0.4.0
accelerate>=0.20.0
pytest>=7.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## 計算リソース要件

- **最小**: GPU 1枚（16GB VRAM）- プロトタイプ実験
- **推奨**: GPU 2-4枚（24GB+ VRAM）- フルスケール実験
- **ストレージ**: 50GB（モデル、データセット、結果）

## 次のステップ

実装計画承認後、以下の順序で進めます：

1. プロジェクト構造の構築
2. FIM計算モジュールの実装とテスト
3. GEVPソルバーの実装とテスト
4. SST-Mergeコアアルゴリズムの実装
5. 小規模実験での動作確認
6. ベースライン実装
7. フルスケール実験の実行
8. 結果分析とレポート作成
