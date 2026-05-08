# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：bidirectional_scan_scheduler.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 11:57 
@Describe：双向扫描调度算法
"""
from typing import List


class BidirectionalScanScheduler:

    def __init__(self, max_batch_size=4, max_compute_usage=1000, max_memory_usage=1000):
        """
        双向扫描调度算法
        :param max_batch_size: 最大任务数
        :param max_compute_usage: GPU计算资源约束
        :param max_memory_usage: GPU显存资源约束
        """
        # batch 最大任务数
        self.max_batch_size = max_batch_size
        # GPU计算资源约束
        self.max_compute_usage = max_compute_usage
        # GPU显存资源约束
        self.max_memory_usage = max_memory_usage


    def can_add_task(self, compute_usage, memory_usage, task):
        """
        通过计算和内存上限来判断任务是否可以加入 batch （暂未启用）
        :param compute_usage: 当前计算资源使用大小
        :param memory_usage: 当前内存资源使用大小
        :param task:当前任务
        :return: True or False
        """
        next_compute = compute_usage + task.comp_cost
        next_memory = memory_usage + task.mem_cost

        if next_compute > self.max_compute_usage:
            return False

        if next_memory > self.max_memory_usage:
            return False

        return True


    def scheduling(self, task_group: List):
        """
        双向扫描调度算法
        :param task_group:
        :return:
        """
        # Step1: 获取任务密度
        density_tasks = []
        for task in task_group:
            density_tasks.append((task, task.comp_density))

        # Step2: 按密度排序，左:memory-intensive，右:compute-intensive
        density_tasks.sort(key=lambda x: x[1])

        # Step2: 双向指针 双向扫描
        left = 0
        right = len(density_tasks) - 1
        batches = []
        while left <= right:
            current_batch = []
            current_compute = 0
            current_memory = 0

            prefer_compute = False  # 当前batch资源偏向，True 偏计算；False 偏内存

            while len(current_batch) < self.max_batch_size and left <= right:
                selected = None
                # 当前batch偏内存，优先加入计算密集型任务
                if not prefer_compute:
                    candidate = density_tasks[right][0]
                    if self.can_add_task(current_compute, current_memory, candidate):
                        selected = candidate
                        right -= 1

                # 当前batch偏计算，优先加入内存密集型任务
                else:
                    candidate = density_tasks[left][0]
                    if self.can_add_task(current_compute, current_memory, candidate):
                        selected = candidate
                        left += 1

                # 无法继续加入任务
                if selected is None:
                    break

                # 加入batch
                current_batch.append(selected)
                current_compute += selected.comp_cost
                current_memory += selected.mem_cost

                # 动态更新batch资源偏向
                if current_compute > current_memory:
                    prefer_compute = True
                else:
                    prefer_compute = False

            # 保存batch
            if len(current_batch) > 0:
                batches.append(current_batch)

        return batches
