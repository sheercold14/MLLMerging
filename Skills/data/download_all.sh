#!/bin/bash
# ==========================================================
# OptMerge VQA & OCR 数据集下载脚本
# 适用于可直连 HuggingFace 的机器
#
# 使用方法:
#   1. pip install huggingface_hub
#   2. huggingface-cli login  (可选，部分数据集需要)
#   3. bash download_all.sh /path/to/save
#
# 默认保存到脚本同目录下的 Optmerge/
# ==========================================================

set -e

BASE_DIR="${1:-$(dirname $0)/Optmerge}"
mkdir -p "$BASE_DIR"/{VQA/{GQA,VQAv2,OKVQA,LLaVA-Instruct,CogVLM-SFT-311K},OCR/{OCRVQA,TextCaps,SynthDoG,LLaVAR,ST-VQA,TextVQA,DocVQA,TabFact},images}

echo "============================================="
echo "OptMerge Dataset Downloader"
echo "Save to: $BASE_DIR"
echo "============================================="

# 检查依赖
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

# ==========================================================
# 1. COCO Images (VQAv2, LLaVA-Instruct 依赖)
# ==========================================================
echo ""
echo "========== COCO Images =========="
echo "VQAv2 和 LLaVA-Instruct 需要 COCO 图像。"
echo "如果你已有 COCO 数据，请跳过此步并手动创建软链接:"
echo "  ln -s /your/coco/path $BASE_DIR/images/coco"
echo ""
read -p "是否下载 COCO train2014+val2014 (~19GB)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p "$BASE_DIR/images/coco"
    echo "Downloading COCO train2014 (~13GB)..."
    wget -c -P "$BASE_DIR/images/coco" http://images.cocodataset.org/zips/train2014.zip
    echo "Downloading COCO val2014 (~6GB)..."
    wget -c -P "$BASE_DIR/images/coco" http://images.cocodataset.org/zips/val2014.zip
    echo "Extracting..."
    unzip -q -n "$BASE_DIR/images/coco/train2014.zip" -d "$BASE_DIR/images/coco/"
    unzip -q -n "$BASE_DIR/images/coco/val2014.zip" -d "$BASE_DIR/images/coco/"
    echo "COCO images done."
fi

# ==========================================================
# 2. VQA 训练数据集 (论文 Table 1, 588K samples)
# ==========================================================
echo ""
echo "========== VQA Datasets =========="

# GQA (en) - 图像内嵌 parquet (~29GB)
download_hf "lmms-lab/GQA" "$BASE_DIR/VQA/GQA"

# VQAv2 (en) - 加载脚本 + 官方 annotations, 图像需 COCO
download_hf "HuggingFaceM4/VQAv2" "$BASE_DIR/VQA/VQAv2"
mkdir -p "$BASE_DIR/VQA/VQAv2/annotations"
echo ">>> [VQAv2] Downloading official annotations..."
wget -q -c -P "$BASE_DIR/VQA/VQAv2/annotations/" https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget -q -c -P "$BASE_DIR/VQA/VQAv2/annotations/" https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
wget -q -c -P "$BASE_DIR/VQA/VQAv2/annotations/" https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget -q -c -P "$BASE_DIR/VQA/VQAv2/annotations/" https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget -q -c -P "$BASE_DIR/VQA/VQAv2/annotations/" https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
for f in "$BASE_DIR/VQA/VQAv2/annotations/"*.zip; do [ -f "$f" ] && unzip -q -n "$f" -d "$BASE_DIR/VQA/VQAv2/annotations/"; done
# 创建 COCO 软链接
if [ -d "$BASE_DIR/images/coco/train2014" ]; then
    ln -sfn "$BASE_DIR/images/coco/train2014" "$BASE_DIR/VQA/VQAv2/train2014"
    ln -sfn "$BASE_DIR/images/coco/val2014" "$BASE_DIR/VQA/VQAv2/val2014"
fi
echo ">>> [VQAv2] annotations done"

# OKVQA (en) - 图像内嵌 parquet (~230MB)
download_hf "Multimodal-Fatima/OK-VQA_train" "$BASE_DIR/VQA/OKVQA"

# LLaVA-Instruct (zh) - JSON 标注, 图像需 COCO (~559MB)
download_hf "liuhaotian/LLaVA-Instruct-150K" "$BASE_DIR/VQA/LLaVA-Instruct"
if [ -d "$BASE_DIR/images/coco/train2014" ]; then
    ln -sfn "$BASE_DIR/images/coco/train2014" "$BASE_DIR/VQA/LLaVA-Instruct/train2014"
fi

# CogVLM Singleround & Multiround (en&zh) - 含图像 (~7.7GB zip)
download_hf "THUDM/CogVLM-SFT-311K" "$BASE_DIR/VQA/CogVLM-SFT-311K"
if [ -f "$BASE_DIR/VQA/CogVLM-SFT-311K/CogVLM-SFT-311K.zip" ]; then
    echo ">>> [CogVLM] Extracting..."
    unzip -q -n "$BASE_DIR/VQA/CogVLM-SFT-311K/CogVLM-SFT-311K.zip" -d "$BASE_DIR/VQA/CogVLM-SFT-311K/"
    echo ">>> [CogVLM] Extracted"
fi

# ==========================================================
# 3. OCR 训练数据集 (论文 Table 1, 238K samples)
# ==========================================================
echo ""
echo "========== OCR Datasets =========="

# OCRVQA (en) - 图像内嵌 parquet
download_hf "howard-hou/OCR-VQA" "$BASE_DIR/OCR/OCRVQA"

# TextCaps (en) - 图像内嵌 parquet (lmms-lab 版本)
download_hf "lmms-lab/TextCaps" "$BASE_DIR/OCR/TextCaps"

# SynthDoG-EN (en) - 合成图像内嵌 parquet
download_hf "naver-clova-ix/synthdog-en" "$BASE_DIR/OCR/SynthDoG"

# LLaVAR (en) - 含图像 zip
download_hf "SALT-NLP/LLaVAR" "$BASE_DIR/OCR/LLaVAR"
if [ -f "$BASE_DIR/OCR/LLaVAR/finetune.zip" ]; then
    echo ">>> [LLaVAR] Extracting finetune.zip..."
    unzip -q -n "$BASE_DIR/OCR/LLaVAR/finetune.zip" -d "$BASE_DIR/OCR/LLaVAR/"
fi
if [ -f "$BASE_DIR/OCR/LLaVAR/pretrain.zip" ]; then
    echo ">>> [LLaVAR] Extracting pretrain.zip..."
    unzip -q -n "$BASE_DIR/OCR/LLaVAR/pretrain.zip" -d "$BASE_DIR/OCR/LLaVAR/"
fi

# ST-VQA (en) - 图像内嵌 parquet (仅 test split)
download_hf "lmms-lab/ST-VQA" "$BASE_DIR/OCR/ST-VQA"

# TextVQA (en) - 图像内嵌 parquet (lmms-lab 版本, ~7GB)
download_hf "lmms-lab/textvqa" "$BASE_DIR/OCR/TextVQA"

# DocVQA (en) - 图像内嵌 parquet
download_hf "lmms-lab/DocVQA" "$BASE_DIR/OCR/DocVQA"

# TabFact (en) - 纯文本表格数据
echo ">>> [TabFact] Cloning from GitHub..."
git clone --depth 1 https://github.com/wenhuchen/Table-Fact-Checking.git "$BASE_DIR/OCR/TabFact/Table-Fact-Checking" 2>/dev/null || echo ">>> [TabFact] already exists or failed"

# ==========================================================
# 4. Summary
# ==========================================================
echo ""
echo "============================================="
echo "Download Summary"
echo "============================================="
echo "--- VQA ---"
du -sh "$BASE_DIR/VQA"/*/ 2>/dev/null
echo ""
echo "--- OCR ---"
du -sh "$BASE_DIR/OCR"/*/ 2>/dev/null
echo ""
echo "--- Total ---"
du -sh "$BASE_DIR"
echo ""
echo "Done! Total estimated: ~100GB"
echo ""
echo "NOTE: DeepForm 和 KLC 为 HF 私有 repo, 已跳过 (OCR 238K 中占比小)"
