# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/7 21:41
@Description: 前缀共享数调度算法（baseline）
"""
from typing import List
from object.batch import Batch
from object.task import Task
from core.scheduler.base_scheduler import BaseScheduler


class PrefixTreeScheduler(BaseScheduler):
    def schedule(self, tasks: List[Task], **kwargs) -> List[Batch]:
        # 1. 构建前缀树 (Radix Tree / Trie)
        tree = self._build_prefix_tree(tasks)

        # 2. 根据 Trie 树拓扑遍历分组
        task_groups = self._group_by_prefix(tree)

        # 3. 打包生成 Batches
        all_batches = []
        max_batch_size = self.config['MAX_SINGLE_BATCH_SIZE']
        for group_id, group in enumerate(task_groups):
            # 将组内任务切分成符合 batch_size 的批次
            for b_idx, i in enumerate(range(0, len(group), max_batch_size)):
                sub_group = group[i:i + max_batch_size]
                all_batches.append(
                    Batch(cluster_id=group_id, batch_id=b_idx, batch_size=len(sub_group), tasks=sub_group))

        return all_batches

    def _build_prefix_tree(self, tasks):
        # 具体的 Trie 树构建逻辑
        pass

    def _group_by_prefix(self, tree):
        # 按树节点切分/合并组逻辑
        pass
