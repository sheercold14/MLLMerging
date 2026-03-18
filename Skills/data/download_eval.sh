#!/bin/bash
# ==========================================================
# OptMerge Evaluation 数据集下载脚本
# 适用于可直连 HuggingFace 的机器
#
# 使用方法:
#   bash download_eval.sh /path/to/save
# ==========================================================

set -e

BASE_DIR="${1:-$(dirname $0)/Optmerge}"
EVAL_DIR="$BASE_DIR/Eval"
mkdir -p "$EVAL_DIR"/{VizWiz,GQA,ChartQA,MathVista,MATH-Vision,TextVQA,OCRVQA,RefCOCO,RefCOCOplus,RefCOCOg,AVQA,MUSIC-AVQA}

echo "============================================="
echo "OptMerge Evaluation Dataset Downloader"
echo "Save to: $EVAL_DIR"
echo "============================================="

if ! command -v huggingface-cli &>/dev/null; then
    echo "Error: huggingface-cli not found. Run: pip install huggingface_hub"
    exit 1
fi

download_hf() {
    local repo=$1
    local target=$2
    local name=$(basename "$target")
    echo ""
    echo ">>> [$name] Downloading $repo ..."
    huggingface-cli download --repo-type dataset "$repo" --local-dir "$target"
    if [ $? -eq 0 ]; then
        echo ">>> [$name] OK ($(du -sh "$target" | cut -f1))"
    else
        echo ">>> [$name] FAILED"
    fi
}

# VQA eval
download_hf "lmms-lab/VizWiz-VQA" "$EVAL_DIR/VizWiz"

# GQA/TextVQA/OCRVQA: 如果训练数据已下载, 创建软链接即可
if [ -d "$BASE_DIR/VQA/GQA" ]; then
    ln -sfn "$BASE_DIR/VQA/GQA" "$EVAL_DIR/GQA"
    echo ">>> [GQA] Symlinked to training data"
else
    download_hf "lmms-lab/GQA" "$EVAL_DIR/GQA"
fi

if [ -d "$BASE_DIR/OCR/TextVQA" ]; then
    ln -sfn "$BASE_DIR/OCR/TextVQA" "$EVAL_DIR/TextVQA"
    echo ">>> [TextVQA] Symlinked to training data"
else
    download_hf "lmms-lab/textvqa" "$EVAL_DIR/TextVQA"
fi

if [ -d "$BASE_DIR/OCR/OCRVQA" ]; then
    ln -sfn "$BASE_DIR/OCR/OCRVQA" "$EVAL_DIR/OCRVQA"
    echo ">>> [OCRVQA] Symlinked to training data"
else
    download_hf "howard-hou/OCR-VQA" "$EVAL_DIR/OCRVQA"
fi

# Chart eval
download_hf "lmms-lab/ChartQA" "$EVAL_DIR/ChartQA"

# Geometry eval
download_hf "AI4Math/MathVista" "$EVAL_DIR/MathVista"
download_hf "MathLLMs/MathVision" "$EVAL_DIR/MATH-Vision"

# Grounding eval
download_hf "lmms-lab/RefCOCO" "$EVAL_DIR/RefCOCO"
download_hf "lmms-lab/RefCOCOplus" "$EVAL_DIR/RefCOCOplus"
download_hf "lmms-lab/RefCOCOg" "$EVAL_DIR/RefCOCOg"

# Audio-Visual eval
download_hf "harryhsing/AVQA-R1-6K" "$EVAL_DIR/AVQA"
echo ">>> [MUSIC-AVQA] Cloning from GitHub..."
git clone --depth 1 https://github.com/GeWu-Lab/MUSIC-AVQA.git "$EVAL_DIR/MUSIC-AVQA/repo" 2>/dev/null || echo ">>> [MUSIC-AVQA] already exists or failed"

echo ""
echo "============================================="
echo "Evaluation Download Summary"
echo "============================================="
du -sh "$EVAL_DIR"/*/ 2>/dev/null
echo ""
echo "--- Total ---"
du -sh "$EVAL_DIR"
