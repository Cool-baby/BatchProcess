# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：main.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 10:22 
@Describe：main
"""
from datetime import datetime
from core.bidirectional_scan_scheduler import BidirectionalScanScheduler
from core.inference_engine import InferenceEngine
from core.prefix_graph import build_prefix_graph
from core.spectral_partition import spectral_partition
from data.running_result_analysis import analysis_tasks_result, analysis_batches_result
from object.batch import Batch
from object.dataset_loader import DatasetLoader
from object.task import generate_task_by_prompt
from utils import save_result

# LLM超参数
MODEL_PATH = "/home/hao/Workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
M_PARAM=7e9  # 大规模模型的参数量
LLM_L=32  # 大规模模型层数
LLM_H=4096  # 每层隐藏维度
LLM_H_KV=128  # KV头特征维度
MAX_NEW_TOKENS = 512  # 最多生成多少个新 token

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
SAMPLE_SIZE = 16  # 每次总共产生多少任务
MAX_SINGLE_BATCH_SIZE = 4  # 最大单批次大小
SCHEDULER = "bilateral"  # 批处理任务调度器

# 实验结果保存路径
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
TASK_LOG_SAVE_PATH = f"./data/task_logs/tasks_logs_{timestamp}.csv"
BATCH_LOG_SAVE_PATH = f"./data/batch_logs/batch_logs_{timestamp}.csv"

# 加载数据集
dataset = DatasetLoader(DATASET_NAME, DATASET_LOCAL_PATH, DATASET_TYPE)

# 抽取 SAMPLE_SIZE 个数量的样本
samples = dataset.extract_data(DATASET_SELECT_SEED, SAMPLE_SIZE)

# 加载LLM推理引擎
inference_worker = InferenceEngine(MODEL_PATH, MAX_NEW_TOKENS)

# 从数据集抽取的数据构建任务
all_tasks = []
for idx, sample in enumerate(samples):
    # 构建任务对象并加入列表
    temp_task = generate_task_by_prompt(DATASET_NAME, DATASET_TYPE, idx, sample)

    if temp_task is not None:
        # tokenizer处理prompt
        ids = inference_worker.get_prompt_token(temp_task.prompt)
        temp_task.input_length = len(ids)
    else:
        print(f"Invalid/blank data")
        continue

    all_tasks.append(temp_task)

print(f"Loaded Tasks: {len(all_tasks)}")

# 预测每个任务输出长度并计算每个任务计算密度
for task in all_tasks:
    # 预测任务输出长度
    task.predicted_output()
    # 计算任务计算密度
    task.cal_computation_density(M_PARAM, LLM_L, LLM_H, LLM_H_KV, SER_GPU_FP, SER_GPU_BANDWIDTH)
    # task.print_task()

# 构建 Prefix Graph
G = build_prefix_graph(all_tasks)
print(f"Prefix graph construction completed, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

# 谱聚类
clusters = spectral_partition(G, num_clusters=SER_NUMBER)
print(f"Spectral clustering completed, total Clusters: {len(clusters)}")

# 初始化任务调度器
scheduler = None
if SCHEDULER == "bilateral":
    scheduler = BidirectionalScanScheduler(MAX_SINGLE_BATCH_SIZE, SER_GPU_FP, SER_GPU_MEMORY)
else:
    raise Exception(f"Scheduler {SCHEDULER} not implemented")

# 并对任务分类形成批次
all_batches = []
for cluster_id, task_group in enumerate(clusters):
    batches = scheduler.scheduling(task_group)
    for batch_id, single_batch in enumerate(batches):
        batch = Batch(cluster_id, batch_id, len(single_batch), single_batch)
        all_batches.append(batch)

print(f"Batch division completed, total batches: {len(all_batches)}")

print("\nBatch processing Start \n")

# 批处理任务
total_latency = 0.0
for batch_id, batch in enumerate(all_batches):
    batch_latency = inference_worker.inference_batch(batch)  # 批处理
    total_latency += batch_latency
    batch.print_batch()

    # 每个批次信息及时保存
    save_result.append_tasks_to_csv(batch.tasks, TASK_LOG_SAVE_PATH)
    save_result.append_batch_to_csv(batch, BATCH_LOG_SAVE_PATH)

print("\nBatch processing End\n")

# 结果分析
analysis_tasks_result(all_tasks)
analysis_batches_result(all_batches)
