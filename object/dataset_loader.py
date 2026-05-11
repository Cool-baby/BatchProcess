# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：dataset_loader.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/11 17:04 
@Describe：数据集加载器
"""
import os
from datasets import load_dataset


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
            self.dataset = load_dataset(self.dataset_local_path, split="train")
        else:
            # 本地不存在则自动下载并保存
            print(f"Download dataset {self.dataset_name} ......")
            self.dataset = load_dataset(self.dataset_local_path, split="train")

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
        # 直接选择前 SAMPLE_SIZE 个
        return selected_datas.select(range(sample_size))
