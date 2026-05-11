# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：batch.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/11 10:30 
@Describe：批次对象，用于批量处理
"""
from typing import List


class Batch:
    def __init__(self, cluster_id, batch_id, batch_size, tasks: List):
        """
        任务批次对象
        :param cluster_id: 簇ID（用于标记不同服务器）
        :param batch_id: 批次ID
        :param batch_size: 批次大小
        :param tasks: 任务列表
        """
        self.cluster_id = cluster_id
        self.batch_id = batch_id
        self.batch_size = batch_size
        self.tasks = tasks

        # batch latency
        self.time_cost = 0

        # token统计
        self.total_generated_tokens = 0
        self.tokens_throughput = 0

        # completion统计
        self.finished_tasks = 0
        self.finished_rate = 0
        self.truncated_tasks = 0
        self.truncated_rate = 0


    def print_batch(self):
        """
        打印当前batch信息
        """
        print(f"Cluster {self.cluster_id}, batch {self.batch_id}, batch size: {self.batch_size} (finished: {self.finished_tasks}, truncated: {self.truncated_tasks}), "
              f"generated tokens: {self.total_generated_tokens}, throughput: {self.tokens_throughput} tokens/sec, "
              f"tasks: {[task.task_id for task in self.tasks]}")
