# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/7 21:41
@Description: 前缀共享数调度算法（baseline）
"""
from typing import List, Dict, Any
from core.scheduler.base_scheduler import BaseScheduler
from object.batch import Batch
from collections import defaultdict
from object.task import Task


class TrieNode:
    """前缀树节点（增加 parent 指针以支持非递归回溯）"""

    def __init__(self, parent=None):
        self.children: Dict[Any, TrieNode] = {}  # token_id -> TrieNode
        self.tasks: List[Any] = []  # 驻留在当前节点的 Task 列表
        self.parent: TrieNode = parent  # 父节点引用


class PrefixTreeScheduler(BaseScheduler):
    """
    前缀树 (Radix/Trie) 调度器 (基线算法)
    迭代版：采用非递归后序遍历，彻底解决 RecursionError 栈溢出问题
    按 ser_number 轮询平分批次到各个服务器节点 (cluster_id)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def schedule(self, tasks: List[Task], **kwargs) -> List[Batch]:
        max_batch_size = self.config.get('MAX_SINGLE_BATCH_SIZE', 8)
        ser_para = self.config.get('SER_PARA', {})
        num_servers = ser_para.get('ser_number', 1)

        # 1. 构建前缀树 (传入 parent 指针)
        root = self._build_prefix_tree(tasks, **kwargs)

        # 2. 非递归自底向上分组
        task_groups = self._group_by_prefix(root, max_batch_size)

        # 3. 轮询分配批次到各服务器，确保各服务器批次数量尽可能相同
        all_batches = []
        server_batch_counts = [0] * num_servers  # 记录各服务器内部的 batch_id

        for idx, group_tasks in enumerate(task_groups):
            # 按 ser_number 循环分配 cluster_id
            cluster_id = idx % num_servers
            batch_id = server_batch_counts[cluster_id]
            server_batch_counts[cluster_id] += 1

            batch = Batch(
                cluster_id=cluster_id,
                batch_id=batch_id,
                batch_size=len(group_tasks),
                tasks=group_tasks
            )
            all_batches.append(batch)

        return all_batches

    def _get_task_tokens(self, task: Task, **kwargs) -> list:
        if hasattr(task, 'prompt_token_ids') and task.prompt_token_ids:
            return task.prompt_token_ids
        elif hasattr(task, 'tokens') and task.tokens:
            return task.tokens

        inference_worker = kwargs.get('inference_worker')
        if inference_worker and hasattr(task, 'prompt'):
            return inference_worker.get_prompt_token(task.prompt)

        return list(getattr(task, 'prompt', ''))

    def _build_prefix_tree(self, tasks: List[Task], **kwargs) -> TrieNode:
        """构建 Token 级别的 Trie 树，并在建树时绑定 parent 引用"""
        root = TrieNode()
        for task in tasks:
            tokens = self._get_task_tokens(task, **kwargs)
            curr = root
            for token in tokens:
                if token not in curr.children:
                    curr.children[token] = TrieNode(parent=curr)
                curr = curr.children[token]
            curr.tasks.append(task)
        return root

    def _group_by_prefix(self, root: TrieNode, max_batch_size: int) -> List[List[Any]]:
        """
        非递归后序遍历（Iterative Post-Order Traversal）
        保障子节点（深层共享前缀）永远在父节点之前被处理
        """
        completed_groups = []

        # 步骤 1：利用双栈/逆序 BFS-DFS 得到严格的后序遍历序列
        s1 = [root]
        s2 = []
        while s1:
            curr = s1.pop()
            s2.append(curr)
            for child in curr.children.values():
                s1.append(child)

        # s2 的逆序 (reversed(s2)) 保证了所有的叶子节点和深层节点都会排在父节点之前处理
        post_order_nodes = reversed(s2)

        # 步骤 2：维护节点未打包任务的暂存区
        node_unmerged_tasks = defaultdict(list)

        for node in post_order_nodes:
            # 汇总当前节点自身携带的任务以及由子节点上传上来的未凑满任务
            curr_tasks = node_unmerged_tasks[node] + list(node.tasks)

            # 只要凑满 max_batch_size 就进行切分打包
            while len(curr_tasks) >= max_batch_size:
                completed_groups.append(curr_tasks[:max_batch_size])
                curr_tasks = curr_tasks[max_batch_size:]

            # 剩余未凑满一个 Batch 的任务，向父节点传递
            if curr_tasks:
                if node.parent is not None:
                    node_unmerged_tasks[node.parent].extend(curr_tasks)
                else:
                    # 根节点剩余的任务（即使不满 max_batch_size）也独立打包
                    completed_groups.append(curr_tasks)

        return completed_groups
