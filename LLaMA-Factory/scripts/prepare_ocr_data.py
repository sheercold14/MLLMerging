"""
Prepare OCR training data for LLaMA-Factory Qwen2-VL LoRA fine-tuning.

Converts various OCR datasets to LLaMA-Factory sharegpt format and registers
them in dataset_info.json.

Available datasets:
- LLaVAR: Text-rich image understanding (19.8K OCR-specific samples)
- TextVQA: Visual QA focused on reading text (parquet -> json conversion)
- DocVQA, OCRVQA, TextCaps, ST-VQA: When downloaded

Usage:
    python scripts/prepare_ocr_data.py --data_root /data/lishichao/data/Optmerge/OCR
"""

import json
import os
import argparse
from pathlib import Path


def prepare_llavar(data_root: str, output_dir: str):
    """Extract OCR-specific samples from LLaVAR (images starting with '1000...')."""
    llavar_dir = os.path.join(data_root, "LLaVAR")
    src_file = os.path.join(llavar_dir, "llava_instruct_150k_llavar_20k.json")

    if not os.path.exists(src_file):
        print(f"[SKIP] LLaVAR not found: {src_file}")
        return None

    with open(src_file) as f:
        data = json.load(f)

    # Filter OCR-specific samples (LLaVAR images start with '1000...')
    ocr_samples = []
    for item in data:
        if item["image"].startswith("1"):
            # Verify image exists
            img_path = os.path.join(llavar_dir, item["image"])
            if os.path.exists(img_path):
                ocr_samples.append({
                    "id": item["id"],
                    "images": [item["image"]],
                    "conversations": item["conversations"],
                })

    output_file = os.path.join(output_dir, "llavar_ocr.json")
    with open(output_file, "w") as f:
        json.dump(ocr_samples, f, ensure_ascii=False, indent=2)

    print(f"[OK] LLaVAR OCR: {len(ocr_samples)} samples -> {output_file}")
    return {
        "llavar_ocr": {
            "file_name": "llavar_ocr.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }


def prepare_textvqa(data_root: str, output_dir: str):
    """Convert TextVQA parquet to LLaMA-Factory sharegpt format."""
    textvqa_dir = os.path.join(data_root, "TextVQA", "data")
    if not os.path.exists(textvqa_dir):
        print(f"[SKIP] TextVQA not found: {textvqa_dir}")
        return None

    try:
        import pandas as pd
        from PIL import Image
        import io
    except ImportError:
        print("[SKIP] TextVQA: pandas/pillow not installed, run after installing dependencies")
        return None

    # Find train parquet files
    train_files = sorted([
        os.path.join(textvqa_dir, f)
        for f in os.listdir(textvqa_dir)
        if f.startswith("train-") and f.endswith(".parquet")
    ])

    if not train_files:
        print("[SKIP] TextVQA: no train parquet files found")
        return None

    # Create image output directory
    img_out_dir = os.path.join(data_root, "TextVQA", "images")
    os.makedirs(img_out_dir, exist_ok=True)

    samples = []
    for pf in train_files:
        print(f"  Processing {os.path.basename(pf)}...")
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            # Save image
            img_id = str(row["image_id"])
            img_filename = f"{img_id}.jpg"
            img_path = os.path.join(img_out_dir, img_filename)
            if not os.path.exists(img_path):
                try:
                    img_data = row["image"]
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img = Image.open(io.BytesIO(img_data["bytes"]))
                    elif isinstance(img_data, Image.Image):
                        img = img_data
                    else:
                        continue
                    img.save(img_path)
                except Exception:
                    continue

            # Use first answer as response
            answers = row.get("answers", [])
            if not answers:
                continue
            answer = answers[0] if isinstance(answers, list) else str(answers)

            samples.append({
                "id": f"textvqa_{row.get('question_id', img_id)}",
                "images": [img_filename],
                "conversations": [
                    {"from": "human", "value": f"<image>\n{row['question']}"},
                    {"from": "gpt", "value": str(answer)},
                ],
            })

    output_file = os.path.join(output_dir, "textvqa_ocr.json")
    with open(output_file, "w") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"[OK] TextVQA: {len(samples)} samples -> {output_file}")
    return {
        "textvqa_ocr": {
            "file_name": "textvqa_ocr.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/lishichao/data/Optmerge/OCR")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for processed data (default: LLaMA-Factory/data/)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    factory_root = os.path.dirname(script_dir)
    output_dir = args.output_dir or os.path.join(factory_root, "data")
    os.makedirs(output_dir, exist_ok=True)

    dataset_info = {}

    # Prepare LLaVAR
    info = prepare_llavar(args.data_root, output_dir)
    if info:
        dataset_info.update(info)

    # Prepare TextVQA (requires pandas)
    info = prepare_textvqa(args.data_root, output_dir)
    if info:
        dataset_info.update(info)

    # Write dataset_info.json
    info_file = os.path.join(output_dir, "dataset_info.json")
    if os.path.exists(info_file):
        with open(info_file) as f:
            existing = json.load(f)
        existing.update(dataset_info)
        dataset_info = existing

    with open(info_file, "w") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] dataset_info.json updated: {info_file}")
    print(f"Registered datasets: {list(dataset_info.keys())}")


if __name__ == "__main__":
    main()
