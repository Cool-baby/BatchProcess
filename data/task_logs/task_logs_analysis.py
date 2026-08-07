# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/7 08:39
@Description: 批处理任务日志数据分析
"""
import pandas as pd
import ast
import numpy as np

def parse_prediction(x):
    try:
        data = ast.literal_eval(x)
        return data.get("median_length", np.nan), data.get("p95_length", np.nan)
    except:
        return np.nan, np.nan

def analyze_log(csv_file):
    df = pd.read_csv(csv_file + ".csv")

    df[["pred_median_length", "pred_p95_length"]] = df["predicted_output_length"].apply(
        lambda x: pd.Series(parse_prediction(x))
    )

    df["median_abs_error"] = abs(df["pred_median_length"] - df["output_length"])
    df["median_relative_error"] = df["median_abs_error"] / (df["output_length"] + 1e-6)

    overall = pd.DataFrame([{
        "task_count": len(df),
        "avg_latency": df["runtime_latency"].mean(),
        "p95_latency": df["runtime_latency"].quantile(0.95),
        "avg_input_length": df["input_length"].mean(),
        "avg_output_length": df["output_length"].mean(),
        "median_MAE": df["median_abs_error"].mean(),
    }])

    print("\nOverall Statistics")
    print(overall.T)

    task_type = df.groupby("task_type").agg(
        task_count=("task_id", "count"),
        avg_latency=("runtime_latency", "mean"),
        p95_latency=("runtime_latency", lambda x: x.quantile(0.95)),
        avg_input_length=("input_length", "mean"),
        avg_output_length=("output_length", "mean"),
        median_MAE=("median_abs_error", "mean"),
    )

    print("\nTask Type Statistics")
    print(task_type)

    overall.to_csv(f"{csv_file}_overall_analysis.csv", index=False)
    task_type.to_csv(f"{csv_file}_task_type_analysis.csv")

if __name__ == "__main__":
    analyze_log("tasks_logs_20260806234909")

