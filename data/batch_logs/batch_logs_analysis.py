# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/7 08:50
@Description: Batch日志文件分析
"""
import os
import re
import glob
import numpy as np
import pandas as pd


def analyze_batch_logs(input_dir, output_file="analysis_summary.csv"):
    """
    读取指定路径下的所有 CSV 批次日志文件，按模型、调度算法和批次大小进行数据统计与汇总。

    参数:
        input_dir (str): CSV 文件所在的文件夹路径
        output_file (str): 分析结果导出的 CSV 文件路径
    """
    # 获取目录下所有的 csv 文件
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    if not csv_files:
        print(f"错误: 在路径 '{input_dir}' 下未找到任何 CSV 文件。")
        return None

    # 正则表达式用于解析文件名
    # 文件名格式: {推理模型}_{调度算法}_{批次大小}_batch_logs_{时间}.csv
    filename_pattern = re.compile(
        r"^(?P<model>.+)_(?P<algorithm>[^_]+)_(?P<batch_size>\d+)_batch_logs_(?P<timestamp>\d+)\.csv$"
    )

    all_data_list = []

    for file_path in csv_files:
        file_name = os.path.basename(file_path)

        # 排除之前生成的结果文件（如果放在同一个目录下）
        if file_name == output_file:
            continue

        match = filename_pattern.match(file_name)
        if not match:
            print(f"跳过不符合命名规范的文件: {file_name}")
            continue

        # 解析文件名中的元数据
        model = match.group("model")
        algorithm = match.group("algorithm")
        target_batch_size = int(match.group("batch_size"))
        timestamp = match.group("timestamp")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"读取文件 {file_name} 失败: {e}")
            continue

        # 校验关键列是否存在
        required_cols = [
            'batch_size', 'time_cost', 'total_generated_tokens',
            'tokens_throughput', 'gpu_memory_usage', 'gpu_usage'
        ]
        if not all(col in df.columns for col in required_cols):
            print(f"警告: 文件 {file_name} 缺少必需的列，已跳过。")
            continue

        # 【核心过滤】剔除不满足批次大小的数据行（例如尾部不完整的批次）
        df_filtered = df[df['batch_size'] == target_batch_size].copy()

        if df_filtered.empty:
            print(f"警告: 文件 {file_name} 在过滤批次大小 ({target_batch_size}) 后无有效数据。")
            continue

        # 附加元信息
        df_filtered['model_name'] = model
        df_filtered['schedule_algorithm'] = algorithm
        df_filtered['target_batch_size'] = target_batch_size
        df_filtered['source_file'] = file_name

        all_data_list.append(df_filtered)

    if not all_data_list:
        print("没有可供分析的有效数据。")
        return None

    # 合并所有文件的有效数据
    full_df = pd.concat(all_data_list, ignore_index=True)

    # 定义自定义计算 P95 的函数
    def p95(series):
        return np.percentile(series, 95)

    # 分组统计 (模型 + 调度算法 + 批次大小)
    grouped = full_df.groupby(['model_name', 'schedule_algorithm', 'target_batch_size'])

    summary_df = grouped.agg(
        valid_batches_count=('batch_size', 'count'),  # 有效批次数量
        avg_latency=('time_cost', 'mean'),  # 平均时延 (s)
        p95_latency=('time_cost', p95),  # P95时延 (s)
        avg_generated_tokens=('total_generated_tokens', 'mean'),  # 平均每个批次生成的token数量
        avg_throughput=('tokens_throughput', 'mean'),  # 平均吞吐量 (tokens/s)
        max_throughput=('tokens_throughput', 'max'),  # 最大吞吐量 (tokens/s)
        min_throughput=('tokens_throughput', 'min'),  # 最小吞吐量 (tokens/s)
        avg_gpu_memory_usage=('gpu_memory_usage', 'mean'),  # 平均GPU显存利用率
        avg_gpu_usage=('gpu_usage', 'mean')  # 平均GPU利用率
    ).reset_index()

    # 对所有浮点数数值列保留两位小数
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    # 排除不需要保留小数的整数列
    float_cols = [col for col in numeric_cols if col not in ['target_batch_size', 'valid_batches_count']]
    summary_df[float_cols] = summary_df[float_cols].round(2)

    # 重命名列名，使其更清晰
    summary_df.rename(columns={
        'model_name': '推理模型',
        'schedule_algorithm': '调度算法',
        'target_batch_size': '批次大小',
        'valid_batches_count': '有效批次数',
        'avg_latency': '平均时延(s)',
        'p95_latency': 'P95时延(s)',
        'avg_generated_tokens': '平均每批次生成的Token数',
        'avg_throughput': '平均吞吐量(Tokens/s)',
        'max_throughput': '最大吞吐量(Tokens/s)',
        'min_throughput': '最小吞吐量(Tokens/s)',
        'avg_gpu_memory_usage': '平均GPU显存使用(GB)',
        'avg_gpu_usage': '平均GPU利用率(%)'
    }, inplace=True)

    # 按照 模型 -> 算法 -> 批次大小 排序
    summary_df = summary_df.sort_values(by=['推理模型', '调度算法', '批次大小']).reset_index(drop=True)

    # 保存结果到 CSV 文件
    summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n分析完成！结果已成功保存至: {output_file}\n")
    print(summary_df.to_string())

    return summary_df


if __name__ == "__main__":
    # 替换为您的日志文件所在目录路径
    LOGS_DIRECTORY = "./"

    # 输出结果文件名
    OUTPUT_CSV_PATH = "bilateral_performance_summary.csv"

    # 执行分析
    analyze_batch_logs(input_dir=LOGS_DIRECTORY, output_file=OUTPUT_CSV_PATH)
