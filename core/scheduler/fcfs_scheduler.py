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
        batches = []
        current_tasks = []

        batch_id = 0
        for task in tasks:
            current_tasks.append(task)
            if len(current_tasks) >= max_batch_size:
                batches.append(
                    Batch(cluster_id=0, batch_id=batch_id, batch_size=len(current_tasks), tasks=current_tasks))
                batch_id += 1
                current_tasks = []

        if current_tasks:
            batches.append(Batch(cluster_id=0, batch_id=batch_id, batch_size=len(current_tasks), tasks=current_tasks))

        return batches
