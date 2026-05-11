# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：save_result.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/9 11:18 
@Describe：保存运行结果
"""
import os
from datetime import datetime
import pandas as pd


def save_tasks_to_csv(tasks, save_path=None):
    """
    保存Task列表到CSV

    :param tasks: List[Task]
    :param save_path: 保存路径
    """
    task_dicts = [task.to_dict() for task in tasks]
    df = pd.DataFrame(task_dicts)
    if save_path is None:
        save_path = "./data/task_logs/tasks_logs_" + datetime.now().strftime("%Y%m%d%H%M%S" + ".csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"Saved {len(tasks)} tasks to {save_path}")


def save_batches_to_csv(batches, save_path=None):
    """
    保存Batch列表到CSV

    :param batches: List[Batch]
    :param save_path: 保存路径
    """
    batch_dicts = [batch.to_dict() for batch in batches]
    df = pd.DataFrame(batch_dicts)
    if save_path is None:
        save_path = "./data/batch_logs/batch_logs_" + datetime.now().strftime("%Y%m%d%H%M%S")+ ".csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"Saved {len(batches)} batches to {save_path}")


def append_tasks_to_csv(tasks, save_path):
    """
    追加保存Task到CSV
    """
    task_dicts = [task.to_dict() for task in tasks]
    df = pd.DataFrame(task_dicts)

    # 文件不存在 -> 写header
    if not os.path.exists(save_path):
        df.to_csv(save_path, index=False, encoding="utf-8-sig", mode="w")
    else:
        # 已存在 -> 追加
        df.to_csv(save_path, index=False, encoding="utf-8-sig", mode="a", header=False)


def append_batch_to_csv(batche, save_path):
    """
    追加保存Batch到CSV
    """
    df = pd.DataFrame([batche.to_dict()])

    if not os.path.exists(save_path):
        df.to_csv(save_path, index=False, encoding="utf-8-sig", mode="w")
    else:
        df.to_csv(save_path, index=False, encoding="utf-8-sig", mode="a", header=False)
