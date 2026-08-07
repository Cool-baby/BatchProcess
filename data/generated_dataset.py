# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/7/28 08:59
@Description: 将从原始数据集抽取的数据文件整理，增加唯一ID，生成有顺序和乱序的数据集，用于测试/训练
"""
import json
import hashlib
import random
from tqdm import tqdm


def generate_sample_id(prompt_id, inference_model):
    """生成唯一sample_id"""
    key = f"{prompt_id}_{inference_model}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


def clean_jsonl(input_file, output_file):
    data = []

    # 读取jsonl并添加sample_id
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            item = json.loads(line)
            item = {
                "sample_id": generate_sample_id(
                    item["prompt_id"],
                    item["inference_model"]
                ),
                **item
            }
            data.append(item)

    print(f"Original samples: {len(data)}")

    # 删除完全重复数据
    unique_data = []
    seen = set()

    for item in tqdm(data, desc="Removing duplicates"):
        duplicate_key = (
            item["prompt_id"],
            item["inference_model"],
            tuple(item["output_token_lengths"])
        )

        if duplicate_key not in seen:
            seen.add(duplicate_key)
            unique_data.append(item)

    print(f"After deduplication: {len(unique_data)}")

    # 按prompt_id和模型排序
    unique_data.sort(key=lambda x: (
        x["prompt_id"],
        x["inference_model"]
    ))

    # 保存数据
    with open(output_file, "w", encoding="utf-8") as f:
        for item in unique_data:
            f.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    print(f"Saved cleaned dataset: {output_file}")


def shuffle_jsonl(input_file, output_file, seed=42):
    """随机打乱jsonl数据集"""
    data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading"):
            data.append(json.loads(line))

    print(f"Original samples: {len(data)}")

    random.seed(seed)
    random.shuffle(data)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc="Saving"):
            f.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

    print(f"Saved shuffled dataset: {output_file}")


if __name__ == "__main__":
    input_file = "token_length_dataset_test.jsonl"
    output_file_sorted = "MoLD_sorted_test.jsonl"
    output_file_shuffled = "MoLD_shuffled_test.jsonl"

    # 数据清洗、去重、排序
    clean_jsonl(input_file, output_file_sorted)

    # 打乱数据用于训练/测试
    shuffle_jsonl(output_file_sorted, output_file_shuffled, seed=42)
