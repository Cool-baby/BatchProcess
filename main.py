# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：main.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 10:22 
@Describe：main
"""
import os
from datasets import load_dataset

from core.bidirectional_scan_scheduler import BidirectionalScanScheduler
from core.inference_engine import InferenceEngine
from core.prefix_graph import build_prefix_graph
from core.spectral_partition import spectral_partition
from object.task import Task


# LLM超参数
MODEL_PATH = "/home/hao/Workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
M_PARAM=7e9  # 大规模模型的参数量
LLM_L=32  # 大规模模型层数
LLM_H=4096  # 每层隐藏维度
LLM_H_KV=128  # KV头特征维度
MAX_NEW_TOKENS = 256  # 最多生成多少个新 token

# 服务器超参数
SER_GPU_FP = 22e12  # 服务节点的浮点计算能力(FLOPS)
SER_GPU_MEMORY = 16 * 1024  # GPU 的最大内存(MB)
SER_GPU_BANDWIDTH = 288e9  # GPU 的内存带宽
SER_NUMBER = 2  # 可用服务器数量，这决定了谱聚类的聚类数量

# 数据集超参数
DATASET_NAME = "allenai/WildChat"  # 数据集名称
DATASET_LOCAL_PATH = "/home/hao/Workspace/datasets/WildChat"  # 数据集本地路径
DATASET_TYPE = "chat"  # 数据集类型
DATASET_SELECT_SEED = 43  # 数据集抽取随机种子

# 批量大小超参数
SAMPLE_SIZE = 200  # 每次总共产生多少任务
MAX_SINGLE_BATCH_SIZE = 4  # 最大单批次大小
SCHEDULER = "bilateral"  # 批处理任务调度器


# 加载数据集
if os.path.exists(DATASET_LOCAL_PATH):
    # 优先加载本地数据集
    dataset = load_dataset(DATASET_LOCAL_PATH, split="train")
    print(f"Load local dataset: {len(dataset)}")
else:
    # 本地不存在则自动下载并保存
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Download dataset: {len(dataset)}")

# 抽取一定数量的样本
dataset = dataset.shuffle(seed=DATASET_SELECT_SEED)
# 直接选择前 SAMPLE_SIZE 个
samples = dataset.select(range(SAMPLE_SIZE))

# 加载LLM推理引擎
inference_worker = InferenceEngine(MODEL_PATH, MAX_NEW_TOKENS)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

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

        # tokenizer处理prompt
        ids = inference_worker.get_prompt_token(prompt)
        task.input_length = len(ids)

        tasks.append(task)
    else:
        raise Exception(f"Dataset {DATASET_NAME} not implemented")

print(f"Loaded Tasks: {len(tasks)}")

# 预测每个任务输出长度并计算每个任务计算密度
for task in tasks:
    # 预测任务输出长度
    task.predicted_output()
    # 计算任务计算密度
    task.cal_computation_density(M_PARAM, LLM_L, LLM_H, LLM_H_KV, SER_GPU_FP, SER_GPU_BANDWIDTH)
    # task.print_task()

# 构建 Prefix Graph
G = build_prefix_graph(tasks)
print(f"Prefix graph construction completed. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# 谱聚类
clusters = spectral_partition(G, num_clusters=SER_NUMBER)
print(f"Spectral clustering completed. Cluster Number: {len(clusters)}")

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

print("\nStart batch processing\n")

# 批处理任务
total_latency = 0.0
for batch_id, batch in enumerate(all_batches):
    batch_latency = inference_worker.inference_batch(batch)
    total_latency += batch_latency
    print(f"[Batch {batch_id}]: latency = {batch_latency:.4f}s")

print(f"Total latency : {total_latency}, Average latency : {total_latency / len(all_batches)}")
