#!/bin/bash
# チャット内で作成したガイドをdocsに保存するスクリプト

BRAIN_DIR="/mnt/iag-02/home/hiromi/.gemini/antigravity/brain/f42523d0-f67c-4666-aa5d-61f5e6c924d6"
DOCS_DIR="docs/sst_merge_guides"

echo "Copying guides from brain directory to docs..."

# 主要なガイドをコピー
cp "$BRAIN_DIR/three_models_experiment_guide.md" "$DOCS_DIR/"
cp "$BRAIN_DIR/full_experiment_guide.md" "$DOCS_DIR/"
cp "$BRAIN_DIR/alignment_analysis.md" "$DOCS_DIR/"
cp "$BRAIN_DIR/final_summary.md" "$DOCS_DIR/"
cp "$BRAIN_DIR/execution_analysis.md" "$DOCS_DIR/"
cp "$BRAIN_DIR/execution_verification.md" "$DOCS_DIR/"

# タスク管理ファイルもコピー（オプション）
cp "$BRAIN_DIR/task.md" "$DOCS_DIR/"
cp "$BRAIN_DIR/walkthrough.md" "$DOCS_DIR/"
cp "$BRAIN_DIR/implementation_plan.md" "$DOCS_DIR/"

echo "✓ All guides copied successfully!"
ls -lh "$DOCS_DIR/"
