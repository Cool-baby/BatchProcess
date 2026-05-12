# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：spectral_partition.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 11:46 
@Describe：谱聚类
"""
import numpy as np
from sklearn.cluster import SpectralClustering


def spectral_partition(prefix_graph, num_clusters=2):
    """
    使用谱聚类对 Prefix Graph 进行划分
    :param prefix_graph: Prefix 相似度图
    :param num_clusters: 聚类数量
    :return: 聚类后的任务列表
    """
    # 获取所有节点
    node_list = list(prefix_graph.nodes())

    # 节点数量
    n = len(node_list)

    # 建立 node -> index 映射
    node_to_idx = {
        node: idx
        for idx, node in enumerate(node_list)
    }

    # 初始化相似度矩阵
    similarity_matrix = np.zeros((n, n))

    # 对角线设为1
    # 表示节点与自身完全相似
    np.fill_diagonal(similarity_matrix, 1.0)

    # 遍历图中的真实边，避免 O(n²) 全遍历
    for u, v, data in prefix_graph.edges(data=True):

        i = node_to_idx[u]
        j = node_to_idx[v]

        # 获取边权重（距离）
        dist = data.get("weight", 1.0)

        # 距离 -> 相似度
        similarity = np.exp(-dist)

        # 无向图对称赋值
        similarity_matrix[i][j] = similarity
        similarity_matrix[j][i] = similarity

    # 给所有节点增加极小弱连接，避免：Graph is not fully connected
    epsilon = 1e-6

    similarity_matrix += epsilon

    # 保持对角线为1
    np.fill_diagonal(similarity_matrix, 1.0)

    # 谱聚类，affinity='precomputed'表示输入的是相似度矩阵
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )

    # 聚类标签
    labels = clustering.fit_predict(similarity_matrix)

    # 根据 label 收集任务
    clusters = {}

    for idx, label in enumerate(labels):

        if label not in clusters:
            clusters[label] = []

        task_id = node_list[idx]

        # 获取原始 task
        task = prefix_graph.nodes[task_id]['task']

        clusters[label].append(task)

    return list(clusters.values())
