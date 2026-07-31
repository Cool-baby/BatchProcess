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


def truncate_context_by_tokens(context_text, input_text, tokenizer, max_prompt_length=2048):
    """
    根据 Token 数量精准截断 Context，确保为 Input 留出足够空间
    :param max_prompt_length: 允许输入给模型的最大 Input Token 数（例如模型窗口是 8k，建议设为 7k 留出安全边际）
    """
    input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
    # 扣除 input 占用的 token，再扣除 50 个 token 给 "Context:\n" 等模板文字留空间
    max_context_tokens = max_prompt_length - len(input_tokens) - 50

    if max_context_tokens <= 0:
        # 极少情况：input 本身就超长了，做基础安全兜底
        return context_text[:1000]

    context_tokens = tokenizer.encode(context_text, add_special_tokens=False)

    # 如果 Context 没有超出限制，直接返回
    if len(context_tokens) <= max_context_tokens:
        return context_text

    # 超长处理：采用“头尾保留法”(保留前 50% 和后 50% 的可用 token)
    half = max_context_tokens // 2
    head_tokens = context_tokens[:half]
    tail_tokens = context_tokens[-half:]

    # 解码回文本，并在中间插入省略提示
    truncated_context = (
            tokenizer.decode(head_tokens, skip_special_tokens=True)
            + "\n\n[... Context truncated due to length limits ...]\n\n"
            + tokenizer.decode(tail_tokens, skip_special_tokens=True)
    )
    return truncated_context


def extract_prompt(selected_datas, tokenizer=None, max_prompt_length=2048):
    """
    从选取的数据集中提出prompt
    :param selected_datas: 从数据集中选取的数据
    :param tokenizer: tokenizer
    :param max_prompt_length: prompt最大长度限制
    :return: prompt数组
    """
    prompts = []
    prompt_id = None

    for sample in selected_datas:
        if "conversation" in sample:
            prompt = sample["conversation"][0]["content"]
        elif "message" in sample:
            prompt = sample["message"][0]["content"]
        elif "question" in sample:  # 适配GSM8K数据集
            prompt = sample["question"]
        elif "prompt" in sample:  # 适配MBPP数据集
            prompt = sample["prompt"]
        elif "context" in sample and "input" in sample:  # LongBench 等长文本数据集
            context_text = sample["context"].strip()
            input_text = sample["input"].strip()
            prompt_id = hashlib.sha256(context_text.encode()).hexdigest()[:16]  # 这样确保相同的LongBench的ID是一致的

            if not input_text:
                prompt = context_text
            else:
                context_text = truncate_context_by_tokens(context_text, input_text, tokenizer, max_prompt_length)
                prompt = f"Context:\n<context>\n{context_text}\n</context>\n\nQuestion:\n{input_text}"  # 通用结构化拼接
        else:
            continue

        prompt = prompt.strip()
        # 获取一个唯一ID
        if prompt_id is None:
            prompt_id = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        # 创建Prompt对象并加入列表
        prompts.append(Prompt(prompt_id, prompt))

    return prompts


class DatasetLoader:
    def __init__(self, dataset_name, dataset_local_path, dataset_type, split_way = "train"):
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
                self.dataset = load_dataset(self.dataset_local_path, split=split_way)
            elif self.dataset_name == "openai/gsm8k":
                self.dataset = load_dataset(self.dataset_local_path, "main", split=split_way)
            elif self.dataset_name == "google-research/mbpp":
                self.dataset = load_dataset("json", data_files=f"{self.dataset_local_path}/sanitized-mbpp.json", split=split_way)
            elif self.dataset_name == "zai-org/LongBench":
                self.dataset = load_dataset("json", data_files=f"{self.dataset_local_path}/data/LongBench_test.jsonl", split=split_way)
                # self.dataset = load_dataset("json", data_files=f"{self.dataset_local_path}/data/2wikimqa.jsonl", split=split_way)
                # self.dataset = load_dataset("json", data_files=f"{self.dataset_local_path}/data/qasper.jsonl", split=split_way)
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
