# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：bidirectional_scan_scheduler.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 11:57 
@Describe：双向扫描调度算法
"""
import logging
from typing import List
from object.batch import Batch
from object.task import Task
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from core.predictor.token_length_predictor import TokenLengthPredictor
from core.scheduler.base_scheduler import BaseScheduler


LAMBDA = 4  # 内存密集型任务超参数
GAMMA = 0.5  # 计算密度风险调整参数，调控Med预测和P95预测权重


class BidirectionalScanScheduler(BaseScheduler):

    def schedule(self, tasks: List[Task], **kwargs) -> List[Batch]:
        """
        基于谱聚类和双向扫描的调度方法
        """
        token_pred = self.config['TOKEN_PREDICTOR']
        model_info = self.config['MODEL_INFO']
        ser_para = self.config['SER_PARA']
        max_batch_size = self.config['MAX_SINGLE_BATCH_SIZE']

        # 1. 加载token输出长度预测器
        logging.info("Loading token length predictor...")
        token_length_predictor = TokenLengthPredictor(token_pred['base_model'], token_pred['checkpoint_path'])

        # 2. 预测每个任务输出长度并计算每个任务计算密度
        logging.info(f"Calculate density for each task...")
        for task in tasks:
            # 预测任务输出长度
            task.predicted_output_length = token_length_predictor.predict_one(task.prompt, model_info['model_name'])
            # 计算任务计算密度
            task.comp_density = task_density(model_info['params'], model_info['num_layers'], model_info['hidden_size'],
                                         model_info['head_dim'], model_info['num_kv_heads'], task.input_length,
                                        task.predicted_output_length, ser_para['gpu_flops'], ser_para['gpu_bandwidth'])
            # task.print_task()

        # 3. 构建 Prefix Graph
        G = build_prefix_graph(tasks)
        logging.info(f"Prefix graph construction completed, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

        # 4. 谱聚类
        clusters = spectral_partition(G, num_clusters=ser_para['ser_number'])
        logging.info(f"Spectral clustering completed, total Clusters: {len(clusters)}")

        # 5. 双向扫描调度算法
        all_batches = []
        for cluster_id, task_group in enumerate(clusters):
            batches = self.bidirectional_scan(task_group, max_batch_size, ser_para['gpu_flops'], ser_para['gpu_memory'])
            for batch_id, single_batch in enumerate(batches):
                batch = Batch(cluster_id, batch_id, len(single_batch), single_batch)
                all_batches.append(batch)

        logging.info(f"Batch division completed, total batches: {len(all_batches)}")

        return all_batches

    def can_add_task(self, compute_usage, memory_usage, task, max_compute_usage, max_memory_usage):
        """
        通过计算和内存上限来判断任务是否可以加入 batch
        :param compute_usage: 当前计算资源使用大小
        :param memory_usage: 当前内存资源使用大小
        :param task: 当前任务
        :param max_compute_usage: GPU计算资源约束
        :param max_memory_usage: GPU显存资源约束
        :return: True or False
        """
        next_compute = compute_usage + task.comp_cost
        next_memory = memory_usage + task.mem_cost

        if next_compute > max_compute_usage:
            return False

        if next_memory > max_memory_usage:
            return False

        return True

    def bidirectional_scan(self, task_group: List, max_batch_size, max_compute_usage, max_memory_usage):
        """
        双向扫描调度算法
        :param task_group: 任务队列
        :param max_batch_size: 单批次最大任务数
        :param max_compute_usage: GPU计算资源约束
        :param max_memory_usage: GPU显存资源约束
        :return: 调度好的批次
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

            while len(current_batch) < max_batch_size and left <= right:
                selected = None
                # 当前batch偏内存，优先加入计算密集型任务
                if not prefer_compute:
                    candidate = density_tasks[right][0]
                    if self.can_add_task(current_compute, current_memory, candidate, max_compute_usage,
                                         max_memory_usage):
                        selected = candidate
                        right -= 1

                # 当前batch偏计算，优先加入内存密集型任务
                else:
                    candidate = density_tasks[left][0]
                    if self.can_add_task(current_compute, current_memory, candidate, max_compute_usage,
                                         max_memory_usage):
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

def task_density(m_param, l, h, h_kv, n_kv, input_length, predicted_output_length, f_ser, bandwidth):
    """
    计算任务密度，较高计算密度对应着计算密集型推理任务,而较低计算密度则表示内存密集型推理任务
    :param m_param: 大规模模型的参数量
    :param l: 大规模模型层数
    :param h: 每层隐藏维度
    :param h_kv: KV头特征维度
    :param n_kv: KV头数量
    :param input_length: 任务输入长度
    :param predicted_output_length: 预测输出长度
    :param f_ser: 服务节点的浮点计算能力
    :param bandwidth: GPU 的内存带宽
    :return: 计算密度
    """
    # 计算以Median为基准的计算密度和内存密度
    comp_dens_med = comp_density(m_param, input_length, predicted_output_length['median_length'], h, l, f_ser)
    mem_dens_med = mem_density(l, h_kv, n_kv, input_length, predicted_output_length['median_length'], bandwidth)
    # 计算以P95为基准的计算密度和内存密度
    comp_dens_p95 = comp_density(m_param, input_length, predicted_output_length['p95_length'], h, l, f_ser)
    mem_dens_p95 = mem_density(l, h_kv, n_kv, input_length, predicted_output_length['p95_length'], bandwidth)

    return (1 - GAMMA) * comp_dens_med / mem_dens_med + GAMMA * comp_dens_p95 / mem_dens_p95


def comp_density(m_param, input_length, predicted_output_length, h, l, f_ser):
    """
    计算密集型任务估算
    """
    return (2 * m_param * (
                input_length + predicted_output_length) + 4 * input_length * input_length * h * l) / f_ser


def mem_density(l, h_kv, n_kv, input_length, predicted_output_length, bandwidth):
    """
    内存密集型任务估算
    """
    return (LAMBDA * l * h_kv * n_kv * (
                input_length * predicted_output_length + 0.5 * predicted_output_length * predicted_output_length)) / bandwidth


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


def common_prefix_length(p1: str, p2: str):
    """
    计算Prompt p1和Prompt p2之间的共享前缀长度
    :param p1: Prompt 1
    :param p2: Prompt 2
    :return: 共享前缀长度
    """
    length = 0

    for c1, c2 in zip(p1, p2):

        if c1 == c2:
            length += 1
        else:
            break

    return length


def prefix_distance(p1: str, p2: str):
    """
    计算Prompt p1和Prompt p2在前缀图中的距离
    :param p1: Prompt 1
    :param p2: Prompt 2
    :return:
    """
    # 计算共享前缀长度
    cp = common_prefix_length(p1, p2)

    # Prompt 1 和 Prompt 2长度
    p1_length = len(p1)
    p2_length = len(p2)

    return 1.0 - cp / max(p1_length, p2_length)


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
