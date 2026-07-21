# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：csv_data_analysis.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/11 18:25 
@Describe： CSV文件数据分析
"""
import pandas as pd
import ast
import numpy as np


csv_path = "results-llama3.2-3B.csv"

# 读取CSV
df = pd.read_csv(csv_path, encoding="utf-8")

print("数据总行数:", len(df))
print("列名:", df.columns.tolist())


# 逐行读取
for index, row in df.iterrows():

    prompt_id = row["prompt_id"]
    prompt = row["prompt_text"]

    # out_put_token 是字符串，需要转换为list
    output_tokens = ast.literal_eval(row["output_token_lengths"])

    # 计算统计量
    median_token = np.median(output_tokens)
    worst_token = np.max(output_tokens)
    std_token = np.std(output_tokens)

    print("=" * 80)
    print("Index:", index)
    print("Prompt ID:", prompt_id)
    print("Prompt:", prompt)

    print("Output Token Array:")
    print(output_tokens)

    print("Statistics:")
    print("Median:", median_token)
    print("Worst:", worst_token)
    print("STD:", std_token)
