# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：bidirectional_scan_scheduler.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 11:57 
@Describe：双向扫描调度算法
"""
from typing import List


def bidirectional_scan_schedule(task_group: List, max_batch_size, max_compute_usage, max_memory_usage):
    density_tasks = []

    for task in task_group:
        # TODO: 将任务计算密度融合过来
        # density = task_density(task)
        density = 0

        density_tasks.append((task, density))

    density_tasks.sort(key=lambda x: x[1])

    # 双向指针
    left = 0
    right = len(density_tasks) - 1

    batches = []

    while left <= right:

        current_batch = []

        current_compute = 0
        current_memory = 0

        # 当前batch资源偏向
        #
        # True:
        #   偏计算
        #
        # False:
        #   偏内存
        #
        prefer_compute = False

        while len(current_batch) < max_batch_size and left <= right:

            selected = None

            # ------------------------------------------------
            # 当前batch偏内存
            #
            # 优先加入计算密集型任务
            # ------------------------------------------------
            if not prefer_compute:

                candidate = density_tasks[right][0]

                if can_add_task(current_batch, current_compute, current_memory, candidate, max_compute_usage, max_memory_usage):
                    selected = candidate
                    right -= 1

            # ------------------------------------------------
            # 当前batch偏计算
            #
            # 优先加入内存密集型任务
            # ------------------------------------------------
            else:

                candidate = density_tasks[left][0]

                if can_add_task(current_batch, current_compute, current_memory, candidate, max_compute_usage, max_memory_usage):
                    selected = candidate
                    left += 1

            # ------------------------------------------------
            # 无法继续加入任务
            # ------------------------------------------------
            if selected is None:
                break

            # 加入batch
            current_batch.append(selected)

            current_compute += selected.comp_cost
            current_memory += selected.mem_cost

            # ------------------------------------------------
            # 动态更新batch资源偏向
            # ------------------------------------------------
            if current_compute > current_memory:
                prefer_compute = True
            else:
                prefer_compute = False

        # 保存batch
        if len(current_batch) > 0:
            batches.append(current_batch)

    return batches

def can_add_task(batch,compute_usage,memory_usage,task, max_compute_usage, max_memory_usage):

    next_compute = compute_usage + task.comp_cost
    next_memory = memory_usage + task.mem_cost

    if next_compute > max_compute_usage:
        return False

    if next_memory > max_memory_usage:
        return False

    return True