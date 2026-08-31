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
from core.inference.inference_engine import InferenceEngine
from core.scheduler.schedule_factory import SchedulerFactory
from utils.running_result_analysis import analysis_tasks_result, analysis_batches_result
from object.task import generate_task_by_prompt
from utils import save_result

# LLM超参数(模型名称、路径、模型总参数量、Transformer block 层数、每层隐藏维度、KV头特征维度)
MODEL_INFO = {"model_name": "Qwen2.5-3B-Instruct", "model_path": "/HOME/scw6e0x/run/zzhao/models/Qwen2.5-3B-Instruct", "params": 3.09e9, "num_layers": 36, "hidden_size": 2048, "num_kv_heads": 2, "head_dim": 128}
# MODEL_INFO = {"model_name": "Llama-3.2-3B-Instruct", "model_path": "/HOME/scw6e0x/run/zzhao/models/Llama-3.2-3B-Instruct", "params": 3.21e9, "num_layers": 28, "hidden_size": 3072, "num_kv_heads": 8, "head_dim": 128}
# MODEL_INFO = {"model_name": "Gemma-3-4b-Instruct", "model_path": "/HOME/scw6e0x/run/zzhao/models/gemma-3-4b-it", "params": 4.00e9, "num_layers": 34, "hidden_size": 2560, "num_kv_heads": 4, "head_dim": 256}

MAX_NEW_TOKENS = 1024  # 最多生成多少个新 token
MAX_INPUT_TOKENS = 2048  # 最多输入多少个 token

# token输出长度预测器本地基础模型及训练好的权重路径
TOKEN_PREDICTOR = {"base_model" : "/HOME/scw6e0x/run/zzhao/models/Qwen2.5-1.5B", "checkpoint_path": "/HOME/scw6e0x/run/zzhao/code/BatchProcess/core/predictor/STABS_prod_model_0806.pt"}

# 服务器超参数(服务器名称，浮点计算能力(FLOPS)，GPU 的最大内存(MB)，GPU 的内存带宽，可用服务器数量（谱聚类聚类数量）)
# SER_PARA = {"ser_name": "4060 Ti", "gpu_flops": 22e12, "gpu_memory": 16 * (1024**3), "gpu_bandwidth": 288 * 1e9, "ser_number": 2}
SER_PARA = {"ser_name": "4090", "gpu_flops": 82.6e12, "gpu_memory": 24 * (1024**3), "gpu_bandwidth": 1008 * 1e9, "ser_number": 1}

DATASET_LOCAL_PATH = "/HOME/scw6e0x/run/zzhao/code/BatchProcess/data/batch_process_test_short.jsonl"  # 数据集本地路径

# 批量大小超参数
SAMPLE_SIZE = 300  # 每次总共产生多少任务
MAX_SINGLE_BATCH_SIZE = 16  # 最大单批次大小
SCHEDULER = "bilateral"  # 批处理任务调度器
# SCHEDULER = "fcfs"  # 批处理任务调度器
# SCHEDULER = "prefix-tree"  # 批处理任务调度器
# SCHEDULER = "sjf"  # 批处理任务调度器
# SCHEDULER = "dftsp"  # 批处理任务调度器
# SCHEDULER = "dftsp"  # 批处理任务调度器
# SCHEDULER = "ranking"  # 批处理任务调度器

# 实验结果保存路径
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
TASK_LOG_SAVE_PATH = f"./data/task_logs/{MODEL_INFO['model_name']}_{SCHEDULER}_{MAX_SINGLE_BATCH_SIZE}_tasks_logs_{timestamp}.csv"
BATCH_LOG_SAVE_PATH = f"./data/batch_logs/{MODEL_INFO['model_name']}_{SCHEDULER}_{MAX_SINGLE_BATCH_SIZE}_batch_logs_{timestamp}.csv"

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

    # 1. 加载数据集
    dataset = load_dataset_from_jsonl(DATASET_LOCAL_PATH)

    # 2. 加载LLM推理引擎
    inference_worker = InferenceEngine(MODEL_INFO['model_name'], MODEL_INFO['model_path'], MAX_NEW_TOKENS, MAX_INPUT_TOKENS)

    # 3. 从数据集抽取的数据构建任务
    logging.info("Data construction task extracted from the dataset...")
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

    # 4. 初始化任务调度器
    logging.info(f"Initialize task scheduler {SCHEDULER}")
    # 调度器需要的信息
    config = {
        'MAX_SINGLE_BATCH_SIZE': MAX_SINGLE_BATCH_SIZE,
        'MODEL_INFO': MODEL_INFO,
        'SER_PARA': SER_PARA,
        'TOKEN_PREDICTOR': TOKEN_PREDICTOR,
    }
    scheduler = SchedulerFactory.create(SCHEDULER, config)

    # 5. 任务调度器调度任务
    logging.info(f"↓↓↓ Start scheduling tasks ↓↓↓ \n")
    all_batches = scheduler.schedule(all_tasks)
    logging.info(f"↑↑↑ Task scheduling completed ↑↑↑ \n")

    # 6. 批处理任务
    logger.info("Batch processing Start")
    for batch_id, batch in enumerate(all_batches):
        inference_worker.inference_batch(batch)  # 批处理
        logger.info(f"Cluster {batch.cluster_id}, batch {batch.batch_id}, batch size: {batch.batch_size},  handle time: {batch.time_cost:.4f} s, "
                    f"avg. GPU memory usage:{batch.gpu_memory_usage:.2f}, avg. GPU usage: {batch.gpu_usage:.2f}, "
                    f"generated tokens: {batch.total_generated_tokens}, throughput: {batch.tokens_throughput:.4f} tk/s, tasks: {[task.task_id for task in batch.tasks]}")

        # 每个批次信息及时保存
        save_result.append_tasks_to_csv(batch.tasks, TASK_LOG_SAVE_PATH)
        save_result.append_batch_to_csv(batch, BATCH_LOG_SAVE_PATH)

    logger.info("Batch processing End")

    # 7. 结果分析
    # analysis_tasks_result(all_tasks)
    # analysis_batches_result(all_batches)


if __name__ == "__main__":
    logger.info("Starting...")
    main()
