# Evaluation Results Summary

Results collected from `VLMEvalKit/outputs/`, using the latest timestamped run for each method.

## Methods & Run Dates

| Method | Latest Run |
|--------|-----------|
| InternVL2_5-1B (baseline) | T20260319 |
| merge_ours_internvl | T20260320 |
| merge_wudi2_exclude_chart | T20260319 |
| merge_wudi2_exclude_geometry | T20260319 |
| merge_wudi2_exclude_grounding | T20260319 |
| merge_wudi2_exclude_ocr | T20260318 |
| merge_wudi2_exclude_vqa | T20260318 |

## Accuracy on Each Benchmark

| Method | TextVQA_VAL | OCRVQA_TESTCORE | VizWiz | GQA_TestDev_Balanced | ChartQA_TEST | MathVista_MINI | MathVision_MINI |
|--------|-------------|-----------------|--------|----------------------|--------------|----------------|-----------------|
| InternVL2_5-1B | | 41.76 | 29.01 | 54.69 | 69.72 | 46.40 | 17.43 |
| merge_ours_internvl | 76.01 | 46.29 | 30.88 | 57.14 | 68.44 | 47.60 | 16.12 |
| merge_wudi2_exclude_chart | | | | | | | |
| merge_wudi2_exclude_geometry | | 46.48 | 31.26 | 57.20 | 70.32 | 44.10 | 16.78 |
| merge_wudi2_exclude_grounding | | 46.52 | 31.04 | 57.24 | 69.24 | 47.00 | 16.78 |
| merge_wudi2_exclude_ocr | | 44.63 | 30.78 | 57.10 | 70.60 | 47.00 | 20.72 |
| merge_wudi2_exclude_vqa | | 46.84 | 30.46 | 55.54 | 70.68 | 47.30 | 17.11 |

> Note: MathVista_MINI and MathVision_MINI scores are judged by gpt-4o-mini. `merge_wudi2_exclude_chart` has no completed evaluation results in its latest run (T20260319).
