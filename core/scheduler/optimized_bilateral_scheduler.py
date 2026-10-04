# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/11 22:20
@Description: 优化原有的双向调度算法，原有的方法基于计算密度+前缀图+谱聚类+双向扫描，但是在任务的密度值都很接近（方差极小），
              双向扫描（Dual Scanner）强行配对首尾任务，不仅无法带来明显的重叠收益，反而打乱了原本顺畅的执行节奏。
              双向扫描策略会强行将一个“计算密集”（通常是长 Prompt + 短 Output）和一个“访存密集”（通常是短 Prompt + 长 Output）
              的请求塞进同一个 Batch。

              核心改进：每次创建新 Batch 时交替从两端选取“锚点任务”，然后穿越整个队列寻找能与其形成资源互补、且长度方差在容忍范围内的请求。
              如果找不到，则允许该 Batch 提前闭合，彻底斩断木桶效应造成的 Padding 浪费。
"""
import logging
import time
from typing import List
import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from object.batch import Batch
from object.task import Task
from core.scheduler.base_scheduler import BaseScheduler

LAMBDA = 4  # 内存密集型任务超参数
GAMMA = 0.5  # 计算密度风险调整参数，调控 Med 预测和 P95 预测权重


class BidirectionalScanSchedulerPro(BaseScheduler):

    def schedule(self, tasks: List[Task], **kwargs) -> List[Batch]:
        """
        基于谱聚类和双向扫描的调度方法（结合服务器间的负载均衡优化）
        """
        if not tasks:
            return []

        token_pred = self.config['TOKEN_PREDICTOR']
        model_info = self.config['MODEL_INFO']
        ser_para = self.config['SER_PARA']
        max_batch_size = self.config['MAX_SINGLE_BATCH_SIZE']
        num_servers = ser_para.get('ser_number', 1)

        # 1. 加载 Token 输出长度预测器
        # logging.info("Loading token length predictor...")
        # load_start = time.perf_counter()
        # token_length_predictor = TokenLengthPredictor(token_pred['base_model'], token_pred['checkpoint_path'])
        # logging.info(f"Loading token predictor completed in {time.perf_counter() - load_start:.4f}s.")

        # 2. 预测每个任务输出长度
        logging.info("Calculate density and resource costs for each task...")
        items = [{"prompt_text": task.prompt, "inference_model": model_info['model_name']} for task in tasks]

        predict_start = time.perf_counter()
        batch_results = token_pred.predict_batch(items, batch_size=16)
        logging.info(f"Token length prediction completed in {time.perf_counter() - predict_start:.4f}s.")

        # 3. 提取模型参数并更新 Task 属性（计算密度、计算开销、内存开销）
        model_params = model_info['params']
        num_layers = model_info['num_layers']
        hidden_size = model_info['hidden_size']
        head_dim = model_info['head_dim']
        num_kv_heads = model_info['num_kv_heads']
        gpu_flops = ser_para['gpu_flops']
        gpu_bandwidth = ser_para['gpu_bandwidth']

        for task, pred in zip(tasks, batch_results):
            task.predicted_output_length = pred

            # 计算计算密度
            task.comp_density = task_density(
                model_params, num_layers, hidden_size,
                head_dim, num_kv_heads, task.input_length,
                task.predicted_output_length, gpu_flops, gpu_bandwidth
            )

            # 动态计算并绑定单一 Task 的物理资源消耗，供 can_add_task 准确评估
            med_len = pred.get('median_length', 1) if isinstance(pred, dict) else int(pred)
            task.comp_cost = comp_density(model_params, task.input_length, med_len, hidden_size, num_layers, gpu_flops)
            task.mem_cost = mem_density(num_layers, head_dim, num_kv_heads, task.input_length, med_len, gpu_bandwidth)

        # 4. 构建 Prefix Graph
        G = build_prefix_graph(tasks)
        logging.info(f"Prefix graph built: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        # 5. 自适应图聚类分组
        if self._should_bypass_clustering(G):
            logging.info("Prefix graph is sparse. Bypassing Spectral Clustering.")
            clusters = [tasks]
        else:
            clusters = spectral_partition(G, num_clusters=num_servers)
            logging.info(f"Spectral clustering completed, Clusters: {len(clusters)}")

        # 6. 双向扫描拆分初步 Batch 清单（暂不分配最终 cluster_id）
        raw_batches = []
        for task_group in clusters:
            batches_in_group = self.bidirectional_scan(
                task_group, max_batch_size, ser_para['gpu_flops'], ser_para['gpu_bandwidth']
            )
            raw_batches.extend(batches_in_group)

        # 7. 负载均衡分配（将 Batch 按计算开销均衡打散到各个 Server/Cluster）
        all_batches = self._balance_batches_across_servers(raw_batches, num_servers)

        logging.info(f"Batch division & load balancing completed, total batches: {len(all_batches)}")
        return all_batches

    def _balance_batches_across_servers(self, raw_batches: List[List[Task]], num_servers: int) -> List[Batch]:
        """
        利用 LPT (Longest Processing Time First) 贪心算法，将 Batch 均衡分配至各个服务器节点
        """
        if not raw_batches:
            return []

        # 计算每个 Batch 的总计算负载
        batch_with_cost = []
        for task_list in raw_batches:
            batch_comp_cost = sum(getattr(t, 'comp_cost', 1.0) for t in task_list)
            batch_with_cost.append((batch_comp_cost, task_list))

        # 按负载从大到小排序
        batch_with_cost.sort(key=lambda x: x[0], reverse=True)

        # 维护每个服务器节点的当前负载和当前已分配的 batch 计数
        server_loads = [0.0] * num_servers
        server_batch_counts = [0] * num_servers
        balanced_batches = []

        for cost, task_list in batch_with_cost:
            # 找到当前总负载最小的服务器节点
            min_server_idx = int(np.argmin(server_loads))

            batch_id = server_batch_counts[min_server_idx]
            batch = Batch(min_server_idx, batch_id, len(task_list), task_list)

            balanced_batches.append(batch)

            # 更新服务器节点负载状态
            server_loads[min_server_idx] += cost
            server_batch_counts[min_server_idx] += 1

        logging.info(f"Server Load Distribution: {dict(enumerate(server_loads))}")
        return balanced_batches

    def _get_task_output_len(self, task: Task) -> int:
        pred = getattr(task, 'predicted_output_length', 1)
        if isinstance(pred, dict):
            return int(pred.get('median_length', pred.get('p95_length', 1)))
        elif isinstance(pred, (int, float)):
            return max(1, int(pred))
        return 1

    def _should_bypass_clustering(self, prefix_graph: nx.Graph) -> bool:
        if prefix_graph.number_of_nodes() <= 1:
            return True
        edges = prefix_graph.edges(data=True)
        if not edges:
            return True

        total_cp = sum(data.get('cp_len', 0) for u, v, data in edges)
        avg_cp = total_cp / len(edges)
        return avg_cp < 20

    def is_length_compatible(self, batch_tasks: List[Task], candidate: Task) -> bool:
        if not batch_tasks:
            return True

        length_tolerance = self.config.get('LENGTH_TOLERANCE', 4.0)
        absolute_tolerance = 256

        cand_in = getattr(candidate, 'input_length', len(getattr(candidate, 'prompt', '')))
        cand_out = self._get_task_output_len(candidate)

        batch_ins = [getattr(t, 'input_length', len(getattr(t, 'prompt', ''))) for t in batch_tasks]
        batch_outs = [self._get_task_output_len(t) for t in batch_tasks]

        max_in = max(max(batch_ins), cand_in)
        min_in = max(1, min(min(batch_ins), cand_in))

        max_out = max(max(batch_outs), cand_out)
        min_out = max(1, min(min(batch_outs), cand_out))

        in_compatible = (max_in / min_in <= length_tolerance) or (max_in - min_in <= absolute_tolerance)
        out_compatible = (max_out / min_out <= length_tolerance) or (max_out - min_out <= absolute_tolerance)

        return in_compatible and out_compatible

    def can_add_task(self, compute_usage, memory_usage, task, max_compute_usage, max_memory_usage):
        next_compute = compute_usage + getattr(task, 'comp_cost', 0)
        next_memory = memory_usage + getattr(task, 'mem_cost', 0)

        if next_compute > max_compute_usage or next_memory > max_memory_usage:
            return False

        return True

    def bidirectional_scan(self, task_group: List, max_batch_size, max_compute_usage, max_memory_usage):
        if not task_group:
            return []

        density_tasks = [(task, task.comp_density) for task in task_group]
        density_tasks.sort(key=lambda x: x[1])

        n = len(density_tasks)
        allocated = set()
        batches = []

        while len(allocated) < n:
            current_batch = []
            current_compute = 0.0
            current_memory = 0.0
            prefer_compute = False

            left, right = 0, n - 1
            while left < n and left in allocated: left += 1
            while right >= 0 and right in allocated: right -= 1

            if left > right: break

            while len(current_batch) < max_batch_size and len(allocated) < n:
                selected_idx = -1
                start, end, step = (right, left - 1, -1) if not prefer_compute else (left, right + 1, 1)

                # Pass 1: 软长度匹配 + 资源约束
                for idx in range(start, end, step):
                    if idx not in allocated:
                        candidate = density_tasks[idx][0]
                        if self.can_add_task(current_compute, current_memory, candidate, max_compute_usage,
                                             max_memory_usage):
                            if self.is_length_compatible(current_batch, candidate):
                                selected_idx = idx
                                break

                # Pass 2: 降级保底，仅约束资源
                if selected_idx == -1:
                    for idx in range(start, end, step):
                        if idx not in allocated:
                            candidate = density_tasks[idx][0]
                            if self.can_add_task(current_compute, current_memory, candidate, max_compute_usage,
                                                 max_memory_usage):
                                selected_idx = idx
                                break

                if selected_idx == -1:
                    break

                selected_task = density_tasks[selected_idx][0]
                current_batch.append(selected_task)
                allocated.add(selected_idx)

                current_compute += getattr(selected_task, 'comp_cost', 0)
                current_memory += getattr(selected_task, 'mem_cost', 0)
                prefer_compute = (current_compute > current_memory)

                while left < n and left in allocated: left += 1
                while right >= 0 and right in allocated: right -= 1

            if current_batch:
                batches.append(current_batch)

        return batches


def task_density(m_param, l, h, h_kv, n_kv, input_length, predicted_output_length, f_ser, bandwidth):
    med_len = predicted_output_length['median_length'] if isinstance(predicted_output_length,
                                                                     dict) else predicted_output_length
    p95_len = predicted_output_length['p95_length'] if isinstance(predicted_output_length,
                                                                  dict) else predicted_output_length

    comp_dens_med = comp_density(m_param, input_length, med_len, h, l, f_ser)
    mem_dens_med = mem_density(l, h_kv, n_kv, input_length, med_len, bandwidth)

    comp_dens_p95 = comp_density(m_param, input_length, p95_len, h, l, f_ser)
    mem_dens_p95 = mem_density(l, h_kv, n_kv, input_length, p95_len, bandwidth)

    mem_dens_med = max(mem_dens_med, 1e-6)
    mem_dens_p95 = max(mem_dens_p95, 1e-6)

    return (1 - GAMMA) * (comp_dens_med / mem_dens_med) + GAMMA * (comp_dens_p95 / mem_dens_p95)


def comp_density(m_param, input_length, predicted_output_length, h, l, f_ser):
    return (2 * m_param * (input_length + predicted_output_length) + 4 * input_length * input_length * h * l) / f_ser


def mem_density(l, h_kv, n_kv, input_length, predicted_output_length, bandwidth):
    return (LAMBDA * l * h_kv * n_kv * (
            input_length * predicted_output_length + 0.5 * predicted_output_length * predicted_output_length)) / bandwidth


def build_prefix_graph(tasks: List[Task]):
    prefix_graph = nx.Graph()
    for task in tasks:
        prefix_graph.add_node(task.task_id, task=task)

    n = len(tasks)
    for i in range(n):
        for j in range(i + 1, n):
            t1 = tasks[i]
            t2 = tasks[j]
            cp = common_prefix_length(t1.prompt, t2.prompt)
            max_len = max(len(t1.prompt), len(t2.prompt))
            dist = 1.0 - (cp / max_len) if max_len > 0 else 0.0
            prefix_graph.add_edge(t1.task_id, t2.task_id, weight=dist, cp_len=cp)

    return prefix_graph


def common_prefix_length(p1: str, p2: str):
    length = 0
    for c1, c2 in zip(p1, p2):
        if c1 == c2:
            length += 1
        else:
            break
    return length


def spectral_partition(prefix_graph, num_clusters=2):
    node_list = list(prefix_graph.nodes())
    n = len(node_list)

    if n <= num_clusters:
        return [[prefix_graph.nodes[node]['task']] for node in node_list]

    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    similarity_matrix = np.zeros((n, n))

    for u, v, data in prefix_graph.edges(data=True):
        i = node_to_idx[u]
        j = node_to_idx[v]
        dist = data.get("weight", 1.0)
        similarity = np.exp(-dist)
        similarity_matrix[i][j] = similarity
        similarity_matrix[j][i] = similarity

    np.fill_diagonal(similarity_matrix, 1.0)

    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )

    labels = clustering.fit_predict(similarity_matrix)

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(prefix_graph.nodes[node_list[idx]]['task'])

    return list(clusters.values())
