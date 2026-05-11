# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：csv_data_analysis.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/11 18:25 
@Describe： CSV文件数据分析
"""
import pandas as pd


def tasks_log_analysis(csv_path):
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 过滤完成的有效任务
    valid_df = df[df["finished"] == True].copy()

    # 平均处理时延（Average Latency）
    avg_latency = valid_df["runtime_latency"].mean()
    print(f"Average Latency: {avg_latency:.4f} s")

    # 计算每秒生成 token 数
    total_tokens = valid_df["output_length"].sum()
    total_time = valid_df["runtime_latency"].sum()
    tokens_per_sec = total_tokens / total_time
    print(f"Throughput: {tokens_per_sec:.2f} tokens/s")

    # 计算系统级吞吐量（Tasks/s）
    task_throughput = len(valid_df) / total_time
    print(f"Task Throughput: {task_throughput:.2f} tasks/s")


if __name__ == "__main__":
    csv_path = "./task_logs/tasks_logs_20260511182223.csv"
    tasks_log_analysis(csv_path)
