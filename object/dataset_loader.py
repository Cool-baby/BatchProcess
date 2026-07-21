# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：dataset_loader.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/11 17:04 
@Describe：数据集加载器
"""
import os
import hashlib
from datasets import load_dataset
from object.prompt import Prompt


def extract_prompt(selected_datas):
    """
    从选取的数据集中提出prompt
    :param selected_datas: 从数据集中选取的数据
    :return: prompt数组
    """
    prompts = []

    for sample in selected_datas:
        if "conversation" in sample:
            prompt = sample["conversation"][0]["content"]
        elif "message" in sample:
            prompt = sample["message"][0]["content"]
        elif "question" in sample:  # 适配GSM8K数据集
            prompt = sample["question"]
        elif "prompt" in sample:  # 适配MBPP数据集
            prompt = sample["prompt"]
        elif "context" in sample:   # 适配LongBench数据集
            prompt = sample["context"]
        else:
            continue

        prompt = prompt.strip()
        # 获取一个唯一ID
        prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        # 创建Prompt对象并加入列表
        prompts.append(Prompt(prompt_id, prompt))

    return prompts


class DatasetLoader:
    def __init__(self, dataset_name, dataset_local_path, dataset_type):
        """
        数据集加载器
        :param dataset_name: 数据集名称
        :param dataset_local_path: 数据集本地路径
        :param dataset_type: 数据集类型
        """
        self.dataset_name = dataset_name
        self.dataset_local_path = dataset_local_path
        self.dataset_type = dataset_type

        self.dataset = None
        # 加载数据集
        if os.path.exists(self.dataset_local_path):
            # 优先加载本地数据集
            if self.dataset_name == "allenai/WildChat":
                self.dataset = load_dataset(self.dataset_local_path, split="train")
            elif self.dataset_name == "openai/gsm8k":
                self.dataset = load_dataset(self.dataset_local_path, "main", split="train")
            elif self.dataset_name == "google-research/mbpp":
                self.dataset = load_dataset("json", data_files=f"{self.dataset_local_path}/sanitized-mbpp.json", split="train")
            elif self.dataset_name == "zai-org/LongBench":
                self.dataset = load_dataset("json", data_files=f"{self.dataset_local_path}/data/multi_news.jsonl", split="train")
            else:
                raise ValueError(f"Unknown dataset {self.dataset_name}.")
        else:
            raise FileNotFoundError("Local dataset not found.")

        if self.dataset is not None:
            print(f"Load local dataset {self.dataset_name} successfully.")


    def extract_data(self, select_seed, sample_size):
        """
        随机提取一定数量的数据
        :param select_seed: 随机种子
        :param sample_size: 提取数量
        :return: 数据列表
        """
        # 抽取一定数量的样本
        selected_datas = self.dataset.shuffle(seed=select_seed)
        # 防止数据集过短导致出现问题
        sample_size = min(sample_size, len(selected_datas))
        # 直接选择前 SAMPLE_SIZE 个
        return selected_datas.select(range(sample_size))
