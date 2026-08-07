# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/7/26 09:20
@Description: 数据集组成成分分析，分析一个数据集中不同推理LLM和不同任务类型的数量
"""
import json
from collections import defaultdict
from tqdm import tqdm


filename = "token_length_dataset_test_EGTP.jsonl"

total_count = 0
model_count = defaultdict(int)
dataset_count = defaultdict(int)
model_dataset_count = defaultdict(lambda: defaultdict(int))


with open(filename, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Analyzing"):
        line = line.strip()
        if not line:
            continue

        data = json.loads(line)

        model = data.get("inference_model")
        dataset = data.get("dataset_name")

        total_count += 1
        model_count[model] += 1
        dataset_count[dataset] += 1
        model_dataset_count[model][dataset] += 1


print(f"Total count: {total_count}\n")

print("=== Dataset Distribution ===")
for dataset, count in dataset_count.items():
    print(f"{dataset}: {count}")

print("\n=== Model Distribution ===")
for model, count in model_count.items():
    print(f"{model}: {count}")

print("\n=== Model-Dataset Distribution ===")
for model, datasets in model_dataset_count.items():
    print(f"\n{model}: {model_count[model]}")
    for dataset, count in datasets.items():
        print(f"  {dataset}: {count}")