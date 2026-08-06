# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：main.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 10:22 
@Describe：main
"""
import json
import logging
import os
from datetime import datetime
import pandas as pd
from core.predictor.token_length_predictor import TokenLengthPredictor
from core.scheduler.bidirectional_scan_scheduler import BidirectionalScanScheduler
from core.inference.inference_engine import InferenceEngine
from core.graph_build.prefix_graph import build_prefix_graph
from core.graph_build.spectral_partition import spectral_partition
from data.running_result_analysis import analysis_tasks_result, analysis_batches_result
from object.batch import Batch
from object.task import generate_task_by_prompt
from utils import save_result

# LLM超参数(模型名称、路径、模型总参数量、Transformer block 层数、每层隐藏维度、KV头特征维度)
MODEL_INFO = {"model_name": "Qwen2.5-3B-Instruct", "model_path": "/home/hao/Workspace/models/qwen/Qwen2.5-3B-Instruct", "params": 3.09e9, "num_layers": 36, "hidden_size": 2048, "num_kv_heads": 128}
# MODEL_INFO = {"model_name": "Qwen2.5-3B-Instruct", "model_path": "/Users/hao/Workspace/Other/models/Qwen2.5-1.5B", "params": 3.09e9, "num_layers": 36, "hidden_size": 2048, "num_kv_heads": 128}
# MODEL_INFO = {"model_name": "Llama-3.2-3B-Instruct", "model_path": "/home/hao/Workspace/models/meta-llama/Llama-3.2-3B-Instruct", "params": 3.21e9, "num_layers": 28, "hidden_size": 3072, "num_kv_heads": 128}
# MODEL_INFO = {"model_name": "Gemma-3-4b-Instruct", "model_path": "/home/hao/Workspace/models/google/gemma-3-4b-it", "params": 4.00e9, "num_layers": 34, "hidden_size": 2560, "num_kv_heads": 256}

MAX_NEW_TOKENS = 1024  # 最多生成多少个新 token
MAX_INPUT_TOKENS = 2048  # 最多输入多少个 token

# token输出长度预测器本地基础模型及训练好的权重路径
TOKEN_PREDICTOR = {"base_model" : "/Users/hao/Workspace/Other/models/Qwen2.5-1.5B", "checkpoint_path": "./core/predictor/"}

# 服务器超参数(服务器名称，浮点计算能力(FLOPS)，GPU 的最大内存(MB)，GPU 的内存带宽，可用服务器数量（谱聚类聚类数量）)
SER_PARA = {"ser_name": "4060 Ti", "gpu_flops": 22e12, "gpu_memory": 16 * 1024, "gpu_bandwidth": 8 * 1024, "ser_number": 2}
# SER_PARA = {"ser_name": "4090", "gpu_flops": 22e12, "gpu_memory": 24 * 1024, "gpu_bandwidth": 8 * 1024, "ser_number": 2}

DATASET_LOCAL_PATH = "./data/batch_process_test.jsonl"  # 数据集本地路径

# 批量大小超参数
SAMPLE_SIZE = 4  # 每次总共产生多少任务
MAX_SINGLE_BATCH_SIZE = 2  # 最大单批次大小
SCHEDULER = "bilateral"  # 批处理任务调度器

# 实验结果保存路径
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
TASK_LOG_SAVE_PATH = f"./data/task_logs/tasks_logs_{timestamp}.csv"
BATCH_LOG_SAVE_PATH = f"./data/batch_logs/batch_logs_{timestamp}.csv"

# 日志与输出目录配置
OUTPUT_DIR = "./data/batch_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(OUTPUT_DIR, f"runlog_{timestamp}.log")

# 配置全局 Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger(__name__)


# 数据加载函数
def load_dataset_from_jsonl(filepath: str) -> pd.DataFrame:
    """安全读取 JSONL 文件，容忍多行换行与格式异常"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据集文件不存在: {filepath}")

    records = []
    skipped_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError as e:
                skipped_count += 1
                logger.warning(f"第 {idx} 行 JSON 解析失败已跳过，错误: {e}")

    logger.info(f"成功读取 {len(records)} 条有效数据 (跳过异常行: {skipped_count})")

    df = pd.DataFrame(records)

    # 字段存在性校验
    required_cols = ['prompt_text', 'dataset_name', 'task_type']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"数据集缺少必要的字段: {missing_cols}")

    return df


def main():

    # 加载数据集
    dataset = load_dataset_from_jsonl(DATASET_LOCAL_PATH)

    # 加载LLM推理引擎
    inference_worker = InferenceEngine(MODEL_INFO['model_name'], MODEL_INFO['model_path'], MAX_NEW_TOKENS, MAX_INPUT_TOKENS)

    # 加载token输出长度预测器
    token_length_predictor = TokenLengthPredictor(TOKEN_PREDICTOR['base_model'], TOKEN_PREDICTOR['checkpoint_path'])

    # 从数据集抽取的数据构建任务
    all_tasks = []
    for _, row in dataset.iterrows():
        if len(all_tasks) >= SAMPLE_SIZE: break
        # 构建任务对象并加入列表
        temp_task = generate_task_by_prompt(_, row['dataset_name'], row['task_type'], row["prompt_text"])

        if temp_task is not None:
            # tokenizer处理prompt
            ids = inference_worker.get_prompt_token(temp_task.prompt)
            temp_task.input_length = len(ids)
        else:
            logger.info(f"Invalid/blank data")
            continue

        all_tasks.append(temp_task)

    logger.info(f"Loaded Tasks: {len(all_tasks)}")

    # 预测每个任务输出长度并计算每个任务计算密度
    for task in all_tasks:
        # 预测任务输出长度
        task.predicted_output_length = token_length_predictor.predict_one(task.prompt, MODEL_INFO['model_name'])
        # 计算任务计算密度
        task.cal_computation_density(MODEL_INFO['params'], MODEL_INFO['num_layers'], MODEL_INFO['hidden_size'], MODEL_INFO['num_kv_heads'], SER_PARA['gpu_flops'], SER_PARA['gpu_bandwidth'])
        # task.print_task()

    # 构建 Prefix Graph
    G = build_prefix_graph(all_tasks)
    logger.info(f"Prefix graph construction completed, nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    # 谱聚类
    clusters = spectral_partition(G, num_clusters=SER_PARA['ser_number'])
    logger.info(f"Spectral clustering completed, total Clusters: {len(clusters)}")

    # 初始化任务调度器
    scheduler = None
    if SCHEDULER == "bilateral":
        scheduler = BidirectionalScanScheduler(MAX_SINGLE_BATCH_SIZE, SER_PARA['gpu_flops'], SER_PARA['gpu_memory'])
    else:
        raise Exception(f"Scheduler {SCHEDULER} not implemented")

    # 并对任务分类形成批次
    all_batches = []
    for cluster_id, task_group in enumerate(clusters):
        batches = scheduler.scheduling(task_group)
        for batch_id, single_batch in enumerate(batches):
            batch = Batch(cluster_id, batch_id, len(single_batch), single_batch)
            all_batches.append(batch)

    logger.info(f"Batch division completed, total batches: {len(all_batches)}")

    logger.info("\nBatch processing Start \n")

    # 批处理任务
    for batch_id, batch in enumerate(all_batches):
        inference_worker.inference_batch(batch)  # 批处理
        batch.print_batch()

        # 每个批次信息及时保存
        save_result.append_tasks_to_csv(batch.tasks, TASK_LOG_SAVE_PATH)
        save_result.append_batch_to_csv(batch, BATCH_LOG_SAVE_PATH)

    logger.info("\nBatch processing End\n")

    # 结果分析
    analysis_tasks_result(all_tasks)
    analysis_batches_result(all_batches)


if __name__ == "__main__":
    main()
