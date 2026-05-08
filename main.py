# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：main.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 10:22 
@Describe：main
"""
import os
import random
from datasets import load_dataset

from core.bidirectional_scan_scheduler import BidirectionalScanScheduler
from core.prefix_graph import build_prefix_graph
from core.spectral_partition import spectral_partition
from object.task import Task


# LLM超参数
M_PARAM=7e9  # 大规模模型的参数量
LLM_L=32  # 大规模模型层数
LLM_H=4096  # 每层隐藏维度
LLM_H_KV=128  # KV头特征维度

# 服务器超参数
SER_GPU_FP = 22e12  # 服务节点的浮点计算能力(FLOPS)
SER_GPU_MEMORY = 16 * 1024  # GPU 的最大内存(MB)
SER_GPU_BANDWIDTH = 288e9  # GPU 的内存带宽

# 数据集超参数
DATASET_NAME = "allenai/WildChat"  # 数据集名称
DATASET_LOCAL_PATH = "./data/WildChat"  # 数据集本地路径
DATASET_TYPE = "chat"  # 数据集类型
DATASET_SELECT_SEED = 42  # 数据集抽取随机种子

# 批量大小超参数
SAMPLE_SIZE = 20  # 每次总共产生多少任务
MAX_SINGLE_BATCH_SIZE = 4  # 最大单批次大小

# 批处理任务调度器
SCHEDULER = "bilateral"

# 加载数据集
if os.path.exists(DATASET_LOCAL_PATH):
    # 优先加载本地数据集
    dataset = load_dataset(DATASET_LOCAL_PATH, split="train")
else:
    # 本地不存在则自动下载并保存
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset.save_to_disk(DATASET_LOCAL_PATH)

# 抽取一定数量的样本
dataset = dataset.shuffle(seed=DATASET_SELECT_SEED)
samples = random.sample(list(dataset), SAMPLE_SIZE)

# 构建任务
tasks = []
for idx, sample in enumerate(samples):

    # 如果是WildChat对话
    if DATASET_NAME == "allenai/WildChat":
        # 提取此对话prompt
        conversation = sample["conversation"]
        if len(conversation) == 0:
            continue
        prompt = conversation[0]["content"]

        # 空prompt过滤
        if prompt is None or len(prompt.strip()) == 0:
            continue

        # 构建任务对象并加入列表
        task = Task(task_id=idx, task_type=DATASET_TYPE, prompt=prompt)
        tasks.append(task)
    else:
        raise Exception(f"Dataset {DATASET_NAME} not implemented")

print(f"Loaded Tasks: {len(tasks)}")

# 预测每个任务输出长度并计算每个任务计算密度
for task in tasks:
    # 预测任务输出长度
    task.predicted_output_length()
    # 计算任务计算密度
    task.cal_computation_density(M_PARAM, LLM_L, LLM_H, LLM_H_KV, SER_GPU_FP, SER_GPU_BANDWIDTH)

# 构建 Prefix Graph
G = build_prefix_graph(tasks)
print("Prefix Graph Built")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# 谱聚类
clusters = spectral_partition(G, num_clusters=2)
print(f"Cluster Number: {len(clusters)}")

# 初始化任务调度器
scheduler = None
if SCHEDULER == "bilateral":
    scheduler = BidirectionalScanScheduler(MAX_SINGLE_BATCH_SIZE, SER_GPU_FP, SER_GPU_MEMORY)
else:
    raise Exception(f"Scheduler {SCHEDULER} not implemented")

# 并对任务分类形成批次
all_batches = []
for cluster_id, task_group in enumerate(clusters):
    print(f"\nCluster {cluster_id}")
    batches = scheduler.scheduling(task_group)

    for batch_id, batch in enumerate(batches):
        print(f"Batch {batch_id}: "f"{[task.task_id for task in batch]}")

    all_batches.extend(batches)

print("\nScheduling Finished")
