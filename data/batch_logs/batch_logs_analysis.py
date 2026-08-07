# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/7 08:50
@Description: Batch日志文件分析
"""
import pandas as pd

def analyze_batch_log(csv_file, output_file="batch_analysis.csv"):
    df = pd.read_csv(csv_file)

    result = {
        "batch_count": len(df),
        "total_tasks": df["batch_size"].sum(),
        "avg_batch_size": df["batch_size"].mean(),

        "avg_time_cost": df["time_cost"].mean(),
        "p95_time_cost": df["time_cost"].quantile(0.95),

        "total_generated_tokens": df["total_generated_tokens"].sum(),
        "avg_generated_tokens_per_batch": df["total_generated_tokens"].mean(),

        "avg_tokens_throughput": df["tokens_throughput"].mean(),
        "max_tokens_throughput": df["tokens_throughput"].max(),
        "min_tokens_throughput": df["tokens_throughput"].min(),

        "avg_finished_tasks": df["finished_tasks"].mean(),
        "avg_truncated_tasks": df["truncated_tasks"].mean()
    }

    result_df = pd.DataFrame([result])

    print(result_df.T)

    result_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )


if __name__ == "__main__":
    batch_file = "batch_logs_20260806234909"
    analyze_batch_log(
        f"{batch_file}.csv",
        f"{batch_file}_analysis.csv"
    )
