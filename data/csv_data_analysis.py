# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：csv_data_analysis.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/11 18:25 
@Describe： CSV文件数据分析
"""
import csv
import json
import ast
import numpy as np
import os

CSV_FILES = [
    # "./raw/results-Llama-3.2-3B-Instruct-WildChat.csv",
    # "./raw/results-Llama-3.2-3B-Instruct-WildChat (new).csv",
    # "./raw/results-Qwen2.5-3B-Instruct-WildChat.csv",
    # "./raw/results-Llama-3.2-3B-Instruct-GSM8K.csv",
    # "./raw/results-Qwen2.5-3B-Instruct-GSM8K.csv",
    # "./raw/results-Llama-3.2-3B-Instruct-LongBench.csv",
    # "./raw/results-Llama-3.2-3B-Instruct-LongBench(qasper).csv",
    # "./raw/results-Qwen2.5-3B-Instruct-LongBench.csv",
]


OUTPUT_JSONL = "token_length_dataset.jsonl"


def process_csv_to_jsonl(input_csv, output_jsonl):
    with open(input_csv, "r", encoding="utf-8-sig") as fin, open(output_jsonl, "a", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        total, skipped = 0, 0
        for idx, row in enumerate(reader):
            try:
                lengths = np.array(ast.literal_eval(row["output_token_lengths"]), dtype=np.float32)
                if len(lengths) == 0:
                    continue

                mean_length, std_length = float(np.mean(lengths)), float(np.std(lengths))

                sample = {
                    "prompt_id": row["prompt_id"],
                    "prompt_text": row["prompt_text"],
                    "dataset_name": row.get("dataset_name", "LongBench"),
                    "task_type": row.get("task_type", "long-sequence"),
                    "inference_model": row["inference_model"],
                    "output_token_lengths": lengths.astype(int).tolist(),
                    "repeat_times": int(row["repeat_times"]),
                    "finished": int(row["finished"]),
                    "statistics": {
                        "median_length": float(np.median(lengths)),
                        "mean_length": mean_length,
                        "p90_length": float(np.percentile(lengths, 90)),
                        "p95_length": float(np.percentile(lengths, 95)),
                        "max_length": float(np.max(lengths)),
                        "min_length": float(np.min(lengths)),
                        "std_length": std_length,
                        "cv_length": float(std_length / mean_length) if mean_length > 0 else 0
                    }
                }

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1

            except Exception as e:
                skipped += 1
                print(f"{input_csv} skip line {idx}: {e}")

    print(f"{input_csv}: processed={total}, skipped={skipped}")


if __name__ == "__main__":
    # 如果之前存在旧文件，先删除，避免重复追加
    # if os.path.exists(OUTPUT_JSONL):
    #     os.remove(OUTPUT_JSONL)

    for csv_file in CSV_FILES:
        process_csv_to_jsonl(csv_file, OUTPUT_JSONL)

    print(f"All finished, saved to {OUTPUT_JSONL}")

