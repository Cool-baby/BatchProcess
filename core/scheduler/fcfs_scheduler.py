# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/7 21:42
@Description: FCFS先来先服务调度算法（baseline）
"""
from typing import List
from object.batch import Batch
from object.task import Task
from core.scheduler.base_scheduler import BaseScheduler


class FCFSScheduler(BaseScheduler):
    def schedule(self, tasks: List[Task], **kwargs) -> List[Batch]:
        max_batch_size = self.config['MAX_SINGLE_BATCH_SIZE']
        ser_para = self.config.get('SER_PARA', {})
        num_servers = ser_para.get('ser_number', 1)

        raw_task_groups = []
        current_tasks = []

        # 1. 经典 FCFS 批次切分
        for task in tasks:
            current_tasks.append(task)
            if len(current_tasks) >= max_batch_size:
                raw_task_groups.append(current_tasks)
                current_tasks = []

        if current_tasks:
            raw_task_groups.append(current_tasks)

        # 2. 轮询分配批次到各服务器 (cluster_id)
        batches = []
        server_batch_counts = [0] * num_servers  # 维护各服务器内部的 batch_id 计数

        for idx, group_tasks in enumerate(raw_task_groups):
            cluster_id = idx % num_servers
            batch_id = server_batch_counts[cluster_id]
            server_batch_counts[cluster_id] += 1

            batch = Batch(
                cluster_id=cluster_id,
                batch_id=batch_id,
                batch_size=len(group_tasks),
                tasks=group_tasks
            )
            batches.append(batch)

        return batches
