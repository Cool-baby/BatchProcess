# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：data_analysis.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/11 10:15 
@Describe：运行结果分析
"""
import pandas as pd

# 任务日志路径
TASK_LOG_PATH = "./task_logs/tasks_logs_20260509130053.csv"


# 读取 CSV
df = pd.read_csv(TASK_LOG_PATH)

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
