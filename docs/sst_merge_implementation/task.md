# SST-Merge実装タスクリスト

## Phase 1: 実験計画の策定
- [x] 既存ドキュメントの分析完了
- [x] 実験設計の策定
  - [x] データセット選定（BeaverTails, MMLU, HumanEval）
  - [x] ベースラインモデル選定（Llama-2-7B/Mistral-7B）
  - [x] 評価指標の定義（安全性、ユーティリティ、Safety Tax）
- [x] 実装計画の作成

## Phase 2: 基盤実装
- [x] プロジェクト構造の構築
- [x] Fisher Information Matrix (FIM) 計算モジュール
  - [x] 基本FIM計算
  - [x] LoRA勾配分散近似
  - [x] VILA原理に基づくパラメータ識別
- [x] GEVP (一般化固有値問題) ソルバー
  - [x] 基本GEVPソルバー
  - [x] 数値安定性の確保
- [x] SST-Mergeコアアルゴリズム
- [x] LoRAユーティリティ

## Phase 3: SST-Mergeコア実装
- [ ] 安全サブスペース選択アルゴリズム
- [ ] LoRAマージング機能
- [ ] 評価フレームワーク

## Phase 4: 実験実行と検証
- [ ] 安全性評価実験
- [ ] ユーティリティ評価実験
- [ ] Safety Tax定量化
- [ ] ベースライン比較（AGL, DARE, TIES-Merging）

## Phase 5: ドキュメント化
- [ ] 実験結果レポート作成
- [ ] コード文書化
