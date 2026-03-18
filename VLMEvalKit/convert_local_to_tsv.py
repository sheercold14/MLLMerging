"""
Convert locally downloaded HuggingFace parquet datasets to VLMEvalKit TSV format.
Plan A: Extract images to files, TSV only contains image_path (absolute paths).

Output structure:
  LMUData/
  ├── ChartQA_TEST.tsv
  ├── images/ChartQA_TEST/*.png
  ├── VizWiz.tsv
  ├── images/VizWiz/*.jpg
  └── ...
"""

import pandas as pd
import os
import json
from tqdm import tqdm

EVAL_ROOT = '/data/lishichao/data/Optmerge/Eval'
OUTPUT_ROOT = '/data/lishichao/data/Optmerge/Eval/LMUData'
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def get_img_dir(dataset_name):
    d = os.path.join(OUTPUT_ROOT, 'images', dataset_name)
    os.makedirs(d, exist_ok=True)
    return d


def save_image(img_bytes, img_dir, filename):
    """Save image bytes to file, return absolute path."""
    path = os.path.join(img_dir, filename)
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(img_bytes)
    return path  # absolute path


def save_tsv(df, name):
    path = os.path.join(OUTPUT_ROOT, f'{name}.tsv')
    df.to_csv(path, sep='\t', index=False)
    print(f'Saved {name}.tsv ({len(df)} rows)')


def convert_chartqa():
    name = 'ChartQA_TEST'
    print(f'Converting {name}...')
    df = pd.read_parquet(f'{EVAL_ROOT}/ChartQA/data/test-00000-of-00001.parquet')
    img_dir = get_img_dir(name)
    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img = row['image']
        fname = img.get('path') or f'{i}.png'
        abs_path = save_image(img['bytes'], img_dir, fname)
        rows.append({
            'index': i,
            'image_path': abs_path,
            'question': row['question'],
            'answer': row['answer'],
        })
    save_tsv(pd.DataFrame(rows), name)


def convert_vizwiz():
    name = 'VizWiz'
    print(f'Converting {name}...')
    data_dir = f'{EVAL_ROOT}/VizWiz/data'
    files = sorted([f for f in os.listdir(data_dir) if f.startswith('val-')])
    dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    img_dir = get_img_dir(name)
    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img = row['image']
        fname = img.get('path') or f'{i}.jpg'
        abs_path = save_image(img['bytes'], img_dir, fname)
        answers = row['answers']
        if isinstance(answers, list):
            answer = answers[0] if answers else ''
        else:
            answer = str(answers)
        rows.append({
            'index': i,
            'image_path': abs_path,
            'question': row['question'],
            'answer': answer,
        })
    save_tsv(pd.DataFrame(rows), name)


def convert_mathvista():
    name = 'MathVista_MINI'
    print(f'Converting {name}...')
    df = pd.read_parquet(f'{EVAL_ROOT}/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet')
    img_dir = get_img_dir(name)
    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_data = row.get('decoded_image') or row.get('image')
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img_bytes = img_data['bytes']
        else:
            img_bytes = img_data
        fname = f'{i}.png'
        abs_path = save_image(img_bytes, img_dir, fname)

        entry = {
            'index': i,
            'image_path': abs_path,
            'question': row['query'] if pd.notna(row.get('query')) else row['question'],
            'answer': str(row['answer']),
            'question_type': row.get('question_type', ''),
            'answer_type': row.get('answer_type', ''),
            'choices': json.dumps(row['choices'].tolist()) if row.get('choices') is not None and (not hasattr(row['choices'], '__len__') or len(row['choices']) > 0) else '',
            'precision': row.get('precision', ''),
            'unit': row.get('unit', '') if pd.notna(row.get('unit')) else '',
        }
        if row.get('metadata'):
            meta = row['metadata']
            if isinstance(meta, dict):
                entry['category'] = meta.get('category', '')
        rows.append(entry)
    save_tsv(pd.DataFrame(rows), name)


def convert_mathvision():
    name = 'MathVision_MINI'
    print(f'Converting {name}...')
    df = pd.read_parquet(f'{EVAL_ROOT}/MATH-Vision/data/testmini-00000-of-00001-f8ff70fcb2f29b1d.parquet')
    img_dir = get_img_dir(name)
    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_data = row.get('decoded_image') or row.get('image')
        if isinstance(img_data, dict) and 'bytes' in img_data:
            img_bytes = img_data['bytes']
        else:
            img_bytes = img_data
        fname = f'{i}.png'
        abs_path = save_image(img_bytes, img_dir, fname)

        options = row.get('options', [])
        entry = {
            'index': i,
            'image_path': abs_path,
            'question': row['question'],
            'answer': str(row['answer']),
            'level': row.get('level', ''),
            'subject': row.get('subject', ''),
        }
        if options is not None and hasattr(options, '__len__') and len(options) > 0:
            for j, opt in enumerate(options):
                entry[chr(65 + j)] = opt
        rows.append(entry)
    save_tsv(pd.DataFrame(rows), name)


def convert_textvqa():
    name = 'TextVQA_VAL'
    print(f'Converting {name}...')
    data_dir = f'{EVAL_ROOT}/TextVQA/TextVQA/data'
    if not os.path.exists(data_dir):
        data_dir = f'{EVAL_ROOT}/TextVQA/data'
    df = pd.read_parquet(data_dir)
    if 'set_name' in df.columns:
        df_val = df[df['set_name'] == 'val'].reset_index(drop=True)
        if len(df_val) == 0:
            df_val = df
    else:
        df_val = df
    print(f'  Found {len(df_val)} val samples')

    img_dir = get_img_dir(name)
    rows = []
    for i, row in tqdm(df_val.iterrows(), total=len(df_val)):
        img = row['image']
        fname = img.get('path') or f'{i}.jpg'
        abs_path = save_image(img['bytes'], img_dir, fname)
        answers = row.get('answers', [])
        if isinstance(answers, list):
            answer = '|||'.join([str(a) for a in answers]) if answers else ''
        else:
            answer = str(answers)
        rows.append({
            'index': i,
            'image_path': abs_path,
            'question': row['question'],
            'answer': answer,
        })
    save_tsv(pd.DataFrame(rows), name)


def convert_ocrvqa():
    print('Converting OCRVQA...')
    data_dir = f'{EVAL_ROOT}/OCRVQA/OCRVQA/data'
    if not os.path.exists(data_dir):
        data_dir = f'{EVAL_ROOT}/OCRVQA/data'
    df = pd.read_parquet(data_dir)
    if 'set_name' in df.columns:
        df_test = df[df['set_name'] == 'test'].reset_index(drop=True)
        if len(df_test) == 0:
            df_test = df
    else:
        df_test = df
    print(f'  Found {len(df_test)} test samples')

    img_dir = get_img_dir('OCRVQA')

    # OCRVQA: multiple questions per image - flatten
    rows = []
    idx = 0
    for img_idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
        img = row['image']
        img_fname = img.get('path') or f'{img_idx}.jpg'
        abs_path = save_image(img['bytes'], img_dir, img_fname)

        questions = row['questions']
        answers = row['answers']
        if isinstance(questions, list):
            for q, a in zip(questions, answers):
                rows.append({
                    'index': idx,
                    'image_path': abs_path,
                    'question': q,
                    'answer': str(a),
                })
                idx += 1
        else:
            rows.append({
                'index': idx,
                'image_path': abs_path,
                'question': str(questions),
                'answer': str(answers),
            })
            idx += 1

    df_out = pd.DataFrame(rows)
    print(f'  Total flattened: {len(df_out)} QA pairs')
    save_tsv(df_out, 'OCRVQA_TEST')

    # TESTCORE subset
    if len(df_out) > 1000:
        df_core = df_out.sample(n=1000, random_state=42).reset_index(drop=True)
        df_core['index'] = range(len(df_core))
        save_tsv(df_core, 'OCRVQA_TESTCORE')
    else:
        save_tsv(df_out, 'OCRVQA_TESTCORE')


def convert_gqa():
    name = 'GQA_TestDev_Balanced'
    print(f'Converting {name}...')
    imgs_df = pd.read_parquet(
        f'{EVAL_ROOT}/GQA/GQA/testdev_balanced_images/testdev-00000-of-00001.parquet'
    )
    instr_df = pd.read_parquet(
        f'{EVAL_ROOT}/GQA/GQA/testdev_balanced_instructions/testdev-00000-of-00001.parquet'
    )

    img_dir = get_img_dir(name)

    # Save all images, build lookup: imageId -> abs path
    img_path_map = {}
    for _, row in tqdm(imgs_df.iterrows(), total=len(imgs_df), desc='Saving images'):
        img = row['image']
        fname = f"{row['id']}.jpg"
        abs_path = save_image(img['bytes'], img_dir, fname)
        img_path_map[row['id']] = abs_path

    rows = []
    for i, row in tqdm(instr_df.iterrows(), total=len(instr_df), desc='Building TSV'):
        img_id = row['imageId']
        if img_id not in img_path_map:
            continue
        rows.append({
            'index': i,
            'image_path': img_path_map[img_id],
            'question': row['question'],
            'answer': row['answer'],
        })
    save_tsv(pd.DataFrame(rows), name)


if __name__ == '__main__':
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else [
        'chartqa', 'vizwiz', 'mathvista', 'mathvision',
        'textvqa', 'ocrvqa', 'gqa'
    ]

    converters = {
        'chartqa': convert_chartqa,
        'vizwiz': convert_vizwiz,
        'mathvista': convert_mathvista,
        'mathvision': convert_mathvision,
        'textvqa': convert_textvqa,
        'ocrvqa': convert_ocrvqa,
        'gqa': convert_gqa,
    }

    for t in targets:
        t = t.lower()
        if t in converters:
            converters[t]()
        else:
            print(f'Unknown dataset: {t}')

    print(f'\nDone! TSV files saved to {OUTPUT_ROOT}')
    print(f'To use: export LMUData={OUTPUT_ROOT}')
