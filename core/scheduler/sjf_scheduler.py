# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/9 22:11
@Description: 最短作业优先 (Shortest Job First, SJF)调度器
"""
from typing import List, Dict, Any
from core.scheduler.base_scheduler import BaseScheduler
from object.batch import Batch
from object.task import Task


class SJFScheduler(BaseScheduler):
    """
    最短作业优先 (Shortest Job First, SJF) 调度器
    按照任务的总 Token 数（输入 Token 数 + 预测输出 Token 数）升序排序，
    优先将短任务打成 Batch 发送，显著降低平均等待时间。
    按 ser_number 轮询平分批次到各个服务器节点 (cluster_id)。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def schedule(self, tasks: List[Task], **kwargs) -> List[Any]:
        if not tasks:
            return []

        max_batch_size = self.config.get('MAX_SINGLE_BATCH_SIZE', 8)
        ser_para = self.config.get('SER_PARA', {})
        num_servers = ser_para.get('ser_number', 1)
        model_name = self.config.get('MODEL_INFO', {}).get('model_name', '')

        # 1. 批量预测 Task 的输出长度 (如果存在 Predictor)
        items = [{"prompt_text": task.prompt, "inference_model": model_name} for task in tasks]
        token_pred = self.config['TOKEN_PREDICTOR']
        batch_results = token_pred.predict_batch(items, batch_size=16)
        for task, pred in zip(tasks, batch_results):
            task.predicted_output_length = pred

        # 2. 估算每个 Task 的作业长度
        for task in tasks:
            input_len = getattr(task, 'input_length', len(getattr(task, 'prompt', '')))
            pred_output = getattr(task, 'predicted_output_length', 0)

            metric_type = "median_length"  # 可根据具体调度算法动态切换

            if isinstance(pred_output, dict):
                # 提取当前模型的预测字典
                model_pred = pred_output.get(model_name, {})
                if isinstance(model_pred, dict):
                    output_len = model_pred.get(metric_type, 0)
                else:
                    output_len = model_pred or 0
            elif isinstance(pred_output, (int, float)):
                output_len = pred_output
            else:
                output_len = 0

            # 保存估算出的总 Workload (确保是数值相加)
            task.estimated_total_tokens = input_len + float(output_len)

        # 3. 核心逻辑：按作业长度升序排序 (SJF)
        sorted_tasks = sorted(tasks, key=lambda t: t.estimated_total_tokens)

        # 4. 贪心切分批次
        raw_task_groups = []
        for i in range(0, len(sorted_tasks), max_batch_size):
            raw_task_groups.append(sorted_tasks[i:i + max_batch_size])

        # 5. 轮询分配批次到各服务器 (cluster_id)，确保各服务器批次数量尽可能相同
        all_batches = []
        server_batch_counts = [0] * num_servers  # 维护各服务器内部的 batch_id 计数

        for idx, group_tasks in enumerate(raw_task_groups):
            cluster_id = idx % num_servers
            batch_id = server_batch_counts[cluster_id]
            server_batch_counts[cluster_id] += 1

            batch = Batch(
                cluster_id=cluster_id,
                batch_id=batch_id,
                batch_size=len(group_tasks),
                tasks=group_tasks
            )
            all_batches.append(batch)

        return all_batches
