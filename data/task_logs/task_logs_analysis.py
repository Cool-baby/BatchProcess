# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/7 08:39
@Description: 批处理任务日志数据分析
"""
import os
import re
import glob
import ast
import numpy as np
import pandas as pd

def parse_predicted_length(val):
    """
    解析 predicted_output_length 列 (例如: "{'median_length': 9.89, 'p95_length': 28.2}")
    安全提取 median_length 和 p95_length。
    """
    if pd.isna(val):
        return np.nan, np.nan
    if isinstance(val, dict):
        return val.get('median_length', np.nan), val.get('p95_length', np.nan)
    try:
        d = ast.literal_eval(str(val))
        if isinstance(d, dict):
            return float(d.get('median_length', np.nan)), float(d.get('p95_length', np.nan))
    except Exception:
        pass
    return np.nan, np.nan


def analyze_task_logs(input_dir, output_file="task_analysis_summary.csv"):
    """
    读取指定路径下的所有 *_tasks_logs_*.csv 文件，解析文件名提取元数据，
    统计任务级评估指标并导出汇总 CSV 文件。
    平均计算密度列 (avg_comp_density) 将保留原始浮点精度。
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"错误: 在路径 '{input_dir}' 下未找到任何 CSV 文件。")
        return None

    # 正则表达式匹配文件名
    # 格式: {推理模型}_{调度算法}_{批次大小}_tasks_logs_{时间}.csv
    filename_pattern = re.compile(
        r"^(?P<model>.+)_(?P<algorithm>[^_]+)_(?P<batch_size>\d+)_tasks_logs_(?P<timestamp>\d+)\.csv$"
    )

    all_tasks = []

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name == output_file:
            continue

        match = filename_pattern.match(file_name)
        if not match:
            print(f"跳过不符合命名规范的文件: {file_name}")
            continue

        model = match.group("model")
        algorithm = match.group("algorithm")
        batch_size = int(match.group("batch_size"))

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"读取文件 {file_name} 失败: {e}")
            continue

        required_cols = ['input_length', 'output_length', 'runtime_latency', 'finished']
        if not all(col in df.columns for col in required_cols):
            print(f"警告: 文件 {file_name} 缺少必需列，已跳过。")
            continue

        # 解析 predicted_output_length
        if 'predicted_output_length' in df.columns:
            parsed = df['predicted_output_length'].apply(parse_predicted_length)
            df['pred_median_length'] = [p[0] for p in parsed]
            df['pred_p95_length'] = [p[1] for p in parsed]
        else:
            df['pred_median_length'] = np.nan
            df['pred_p95_length'] = np.nan

        # MAE 与 MAPE
        df['pred_median_mae'] = (df['pred_median_length'] - df['output_length']).abs()
        df['pred_median_mape'] = np.where(
            df['output_length'] > 0,
            (df['pred_median_mae'] / df['output_length']) * 100,
            np.nan
        )

        # 比例与速率
        df['io_ratio'] = np.where(df['output_length'] > 0, df['input_length'] / df['output_length'], np.nan)
        df['token_gen_rate'] = np.where(df['runtime_latency'] > 0, df['output_length'] / df['runtime_latency'], np.nan)

        # 元数据
        df['model_name'] = model
        df['schedule_algorithm'] = algorithm
        df['batch_size_config'] = batch_size

        all_tasks.append(df)

    if not all_tasks:
        print("没有可供分析的有效任务数据。")
        return None

    full_df = pd.concat(all_tasks, ignore_index=True)

    def p95(series):
        valid = series.dropna()
        return np.percentile(valid, 95) if len(valid) > 0 else np.nan

    def p99(series):
        valid = series.dropna()
        return np.percentile(valid, 99) if len(valid) > 0 else np.nan

    grouped = full_df.groupby(['model_name', 'schedule_algorithm', 'batch_size_config'])

    # 动态构建聚合规则
    agg_dict = {
        'total_tasks': ('finished', 'count'),
        'completed_tasks': ('finished', lambda x: (x == 1).sum()),
        'completion_rate': ('finished', lambda x: (x == 1).mean() * 100),
        'avg_latency': ('runtime_latency', 'mean'),
        'p95_latency': ('runtime_latency', p95),
        'p99_latency': ('runtime_latency', p99),
        'avg_input_len': ('input_length', 'mean'),
        'avg_output_len': ('output_length', 'mean'),
        'avg_io_ratio': ('io_ratio', 'mean'),
        'pred_median_mae': ('pred_median_mae', 'mean'),
        'pred_median_mape': ('pred_median_mape', 'mean'),
        'avg_gen_rate': ('token_gen_rate', 'mean'),
    }

    if 'comp_density' in full_df.columns:
        agg_dict['avg_comp_density'] = ('comp_density', 'mean')
    if 'comp_cost' in full_df.columns:
        agg_dict['avg_comp_cost'] = ('comp_cost', 'mean')
    if 'mem_cost' in full_df.columns:
        agg_dict['avg_mem_cost'] = ('mem_cost', 'mean')

    summary_df = grouped.agg(**agg_dict).reset_index()

    # 【数值精度处理】排除计算密度列，不强制保留两位小数
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['batch_size_config', 'total_tasks', 'completed_tasks', 'avg_comp_density']
    float_cols_to_round = [c for c in numeric_cols if c not in exclude_cols]
    summary_df[float_cols_to_round] = summary_df[float_cols_to_round].round(2)

    rename_dict = {
        'model_name': '推理模型',
        'schedule_algorithm': '调度算法',
        'batch_size_config': '批次大小',
        'total_tasks': '总任务数',
        'completed_tasks': '完成任务数',
        'completion_rate': '任务完成率(%)',
        'avg_latency': '平均延迟(s)',
        'p95_latency': 'P95延迟(s)',
        'p99_latency': 'P99延迟(s)',
        'avg_input_len': '平均输入长度',
        'avg_output_len': '平均输出长度',
        'avg_io_ratio': '平均输入/输出比',
        'pred_median_mae': '预测Length_MAE',
        'pred_median_mape': '预测Length_MAPE(%)',
        'avg_gen_rate': '单任务生成速率(Tokens/s)',
        'avg_comp_density': '平均计算密度',
        'avg_comp_cost': '平均计算开销',
        'avg_mem_cost': '平均内存开销'
    }

    summary_df.rename(columns={k: v for k, v in rename_dict.items() if k in summary_df.columns}, inplace=True)
    summary_df.sort_values(by=['推理模型', '调度算法', '批次大小'], inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n分析完成！任务评估结果已保存至: {output_file}\n")
    print(summary_df.to_string())

    return summary_df


if __name__ == "__main__":
    # 替换为您的 tasks_logs 文件夹路径
    LOGS_DIRECTORY = "./"
    OUTPUT_CSV_PATH = "task_bilateral_performance_summary.csv"

    analyze_task_logs(input_dir=LOGS_DIRECTORY, output_file=OUTPUT_CSV_PATH)
