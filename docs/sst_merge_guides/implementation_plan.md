# Phase 8-10実装計画

## 概要

Phase 8-10では、評価パイプライン、ベンチマーク、エンドツーエンド統合を実装します。これによりSST-Mergeの完全なパイプラインが完成します。

## 現状確認

✅ **既存の実装**:
- [`src/evaluation/safety_evaluator.py`](file:///mnt/iag-02/home/hiromi/src/SST_merge/src/evaluation/safety_evaluator.py): `SafetyEvaluator`クラス
- [`src/evaluation/utility_evaluator.py`](file:///mnt/iag-02/home/hiromi/src/SST_merge/src/evaluation/utility_evaluator.py): `UtilityEvaluator`クラス
- [`src/evaluation/metrics_reporter.py`](file:///mnt/iag-02/home/hiromi/src/SST_merge/src/evaluation/metrics_reporter.py): `MetricsReporter`クラス

### 既存の評価モジュール

**SafetyEvaluator**:
- `evaluate_refusal_rate()`: 有害な指示に対する拒否率
- `evaluate_jailbreak_resistance()`: Jailbreak攻撃への耐性

**UtilityEvaluator**:
- `evaluate_mmlu()`: MMLU精度（推論能力）
- `evaluate_humaneval()`: HumanEval Pass@1（コーディング能力）

**MetricsReporter**:
- `compute_composite_score()`: 複合スコア計算
- `compute_pareto_distance()`: パレート距離計算
- `visualize_safety_utility_tradeoff()`: 可視化
- `generate_report()`: レポート生成

## 提案する変更

### Phase 8: scripts/test_evaluation.py [NEW]

評価パイプラインのテストスクリプト:

```python
#!/usr/bin/env python3
"""
Phase 8: 評価パイプラインのテストスクリプト

テスト内容:
- SafetyEvaluatorのテスト
- UtilityEvaluatorのテスト
- MetricsReporterのテスト
"""
```

### Phase 9: scripts/test_benchmark.py [NEW]

ベンチマークスクリプト:

```python
#!/usr/bin/env python3
"""
Phase 9: ベンチマークスクリプト

テスト内容:
- 複数手法の比較
- パレート効率分析
- レポート生成
"""
```

### Phase 10: scripts/test_end_to_end.py [NEW]

エンドツーエンド統合スクリプト:

```python
#!/usr/bin/env python3
"""
Phase 10: エンドツーエンド統合テスト

テスト内容:
- Phase 1-10の完全なパイプライン
- LoRAダウンロードからレポート生成まで
"""
```

## 検証計画

### 自動テスト

```bash
# Phase 8
python scripts/test_evaluation.py

# Phase 9
python scripts/test_benchmark.py

# Phase 10
python scripts/test_end_to_end.py
```

**期待される結果**:
- ✓ Phase 8: 評価パイプラインテスト成功
- ✓ Phase 9: ベンチマークテスト成功
- ✓ Phase 10: エンドツーエンドテスト成功
