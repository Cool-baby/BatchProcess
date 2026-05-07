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
    node_list = list(prefix_graph.nodes())

    n = len(node_list)

    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if i == j:
                similarity_matrix[i][j] = 1.0

            else:

                edge = prefix_graph.get_edge_data(node_list[i], node_list[j])

                dist = edge['weight']

                similarity = 1.0 - dist

                similarity_matrix[i][j] = similarity

    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans', random_state=42)

    labels = clustering.fit_predict(similarity_matrix)

    # 聚类结果
    clusters = {}

    for idx, label in enumerate(labels):

        if label not in clusters:
            clusters[label] = []

        task_id = node_list[idx]

        task = prefix_graph.nodes[task_id]['task']

        clusters[label].append(task)

    return list(clusters.values())
