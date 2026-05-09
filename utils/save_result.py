# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：save_result.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/9 11:18 
@Describe：保存运行结果
"""
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
