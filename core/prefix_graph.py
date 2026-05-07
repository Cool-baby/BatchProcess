# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess
@File    ：prefix_graph.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 11:40
@Describe：前缀图
"""
import networkx as nx
from typing import List

from core.prompt_distance import prefix_distance
from object.task import Task


def build_prefix_graph(tasks: List[Task]):
    """
    构建Prompt graph
    :param tasks: 任务列表
    :return: Prompt graph
    """
    # 初始化图
    prefix_graph = nx.Graph()

    # 添加节点
    for task in tasks:
        prefix_graph.add_node(task.task_id, task=task)

    # 添加边
    n = len(tasks)

    for i in range(n):
        for j in range(i + 1, n):
            t1 = tasks[i]
            t2 = tasks[j]

            dist = prefix_distance(t1.prompt, t2.prompt)

            prefix_graph.add_edge(t1.task_id, t2.task_id, weight=dist)

    return prefix_graph
