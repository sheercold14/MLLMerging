# -*- coding: utf-8 -*-
import os
import sys
import csv

def extract_summary_score(file_path):
    name = os.path.basename(file_path).lower()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rows = list(csv.reader(f))
    except Exception:
        return None, None

    if len(rows) < 2:
        return None, None

    if 'textvqa' in name:
        return 'TextVQA_VAL', rows[1][0]
    if 'ocrvqa_testcore' in name:
        return 'OCRVQA_TESTCORE', rows[1][0]
    if 'vizwiz' in name:
        return 'VizWiz', rows[1][0]
    if 'gqa_testdev_balanced' in name:
        return 'GQA_TestDev_Balanced', rows[1][0]
    if 'chartqa_test' in name:
        return 'ChartQA_TEST', rows[1][-1]
    if 'mathvista' in name and 'score.csv' in name:
        judge = None
        marker = '_mathvista_mini_'
        if marker in name:
            judge = name.split(marker, 1)[1].replace('_score.csv', '')
        for row in rows[1:]:
            if row and row[0].strip().lower() == 'overall':
                key = 'MathVista_MINI'
                if judge:
                    key = f'{key} [{judge}]'
                return key, row[-1]
    if 'mathvision' in name and 'score.csv' in name:
        judge = None
        marker = '_mathvision_mini_'
        if marker in name:
            judge = name.split(marker, 1)[1].replace('_score.csv', '')
        for row in rows[1:]:
            if row and row[0].strip().lower() == 'overall':
                key = 'MathVision_MINI'
                if judge:
                    key = f'{key} [{judge}]'
                return key, row[-1]
    return None, None


def print_final_summary(folder):
    summary = {}
    for root, dirs, files in os.walk(folder):
        for filename in files:
            lowered = filename.lower()
            if 'acc' not in lowered and 'score' not in lowered:
                continue
            file_path = os.path.join(root, filename)
            if os.path.islink(file_path):
                continue
            key, value = extract_summary_score(file_path)
            if key is not None and value is not None:
                summary[key] = value

    if not summary:
        return

    order = [
        'TextVQA_VAL',
        'OCRVQA_TESTCORE',
        'VizWiz',
        'GQA_TestDev_Balanced',
        'ChartQA_TEST',
    ]
    print('Final Scores:')
    for key in order:
        if key in summary:
            print(f'{key}: {summary[key]}')
    extra_keys = sorted(
        k for k in summary
        if k not in order
    )
    for key in extra_keys:
        print(f'{key}: {summary[key]}')
    print('-' * 50)


def process_mathvista_content(content):
    target_names = {"geometry reasoning", "algebraic reasoning", "geometry problem solving"}
    values = []
    try:
        reader = csv.reader(content.splitlines())
        headers = next(reader, None)
        for row in reader:
            if row and row[0].strip().lower() in target_names:
                try:
                    value = float(row[-1].strip().strip('"'))
                    values.append(value)
                except Exception:
                    pass
    except Exception:
        pass
    if values:
        return sum(values) / len(values)
    return None

def process_mathvision_content(content):
    target_names = {"metric geometry - angle", "metric geometry - area", "metric geometry - length", "solid geometry"}
    values = []
    try:
        reader = csv.reader(content.splitlines())
        for row in reader:
            if row and row[0].strip().lower() in target_names:
                try:
                    value = float(row[-1].strip().strip('"'))
                    values.append(value)
                except Exception:
                    pass
    except Exception:
        pass
    if values:
        return sum(values) / len(values)
    return None

def search_and_print(folder):
    for root, dirs, files in os.walk(folder):
        for filename in files:
            if 'acc' in filename.lower() or 'score' in filename.lower():
                file_path = os.path.join(root, filename)
                if os.path.islink(file_path):
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    content = f'Error reading file: {e}'
                print(f'File: {file_path}')
                print('Content:')
                print(content)
                if 'mathvista' in filename.lower():
                    avg = process_mathvista_content(content)
                    if avg is not None:
                        print(f'Average target score: {avg}')
                    else:
                        print('Target rows not found or parse failed')
                if 'mathvision' in filename.lower():
                    avg = process_mathvision_content(content)
                    if avg is not None:
                        print(f'Average target score: {avg}')
                    else:
                        print('Target rows not found or parse failed')
                print('-' * 50)

if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    print_final_summary(folder)
    search_and_print(folder)
