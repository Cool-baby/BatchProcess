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


def analyze_task_logs(input_dir,
                      overall_output="task_overall_summary.csv",
                      by_type_output="task_by_type_summary.csv"):
    """
    读取任务日志 CSV 文件，按模型、调度算法、批次大小以及【任务类型 task_type】进行双维度统计。

    参数:
        input_dir (str): CSV 日志所在文件夹路径
        overall_output (str): 全局汇总表保存路径
        by_type_output (str): 按任务类型细分的汇总表保存路径
    """
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"错误: 在路径 '{input_dir}' 下未找到任何 CSV 文件。")
        return None, None

    # 正则匹配文件名: {推理模型}_{调度算法}_{批次大小}_tasks_logs_{时间}.csv
    filename_pattern = re.compile(
        r"^(?P<model>.+)_(?P<algorithm>[^_]+)_(?P<batch_size>\d+)_tasks_logs_(?P<timestamp>\d+)\.csv$"
    )

    all_tasks = []

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name in [overall_output, by_type_output]:
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

        # 如果没有 task_type 列，则填充为 'unknown'
        if 'task_type' not in df.columns:
            df['task_type'] = 'unknown'

        # 解析 predicted_output_length 字段
        if 'predicted_output_length' in df.columns:
            parsed = df['predicted_output_length'].apply(parse_predicted_length)
            df['pred_median_length'] = [p[0] for p in parsed]
            df['pred_p95_length'] = [p[1] for p in parsed]
        else:
            df['pred_median_length'] = np.nan
            df['pred_p95_length'] = np.nan

        # 衍生计算列
        df['pred_median_mae'] = (df['pred_median_length'] - df['output_length']).abs()
        df['pred_median_mape'] = np.where(
            df['output_length'] > 0,
            (df['pred_median_mae'] / df['output_length']) * 100,
            np.nan
        )

        df['io_ratio'] = np.where(df['output_length'] > 0, df['input_length'] / df['output_length'], np.nan)
        df['token_gen_rate'] = np.where(df['runtime_latency'] > 0, df['output_length'] / df['runtime_latency'], np.nan)

        # 附加文件名提取出的元数据
        df['model_name'] = model
        df['schedule_algorithm'] = algorithm
        df['batch_size_config'] = batch_size

        all_tasks.append(df)

    if not all_tasks:
        print("没有可供分析的有效任务数据。")
        return None, None

    full_df = pd.concat(all_tasks, ignore_index=True)

    # 统计辅助函数
    def p95(series):
        valid = series.dropna()
        return np.percentile(valid, 95) if len(valid) > 0 else np.nan

    def p99(series):
        valid = series.dropna()
        return np.percentile(valid, 99) if len(valid) > 0 else np.nan

    # 通用聚合逻辑字典
    base_agg_dict = {
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
        base_agg_dict['avg_comp_density'] = ('comp_density', 'mean')
    if 'comp_cost' in full_df.columns:
        base_agg_dict['avg_comp_cost'] = ('comp_cost', 'mean')
    if 'mem_cost' in full_df.columns:
        base_agg_dict['avg_mem_cost'] = ('mem_cost', 'mean')

    rename_dict = {
        'model_name': '推理模型',
        'schedule_algorithm': '调度算法',
        'batch_size_config': '批次大小',
        'task_type': '任务类型',
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

    # 格式化小数处理函数（保持 avg_comp_density 高精度，其他保留 2 位小数）
    def process_precision_and_rename(df, is_by_type=False):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['batch_size_config', 'total_tasks', 'completed_tasks', 'avg_comp_density']
        float_cols_to_round = [c for c in numeric_cols if c not in exclude_cols]
        df[float_cols_to_round] = df[float_cols_to_round].round(2)

        df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns}, inplace=True)
        sort_cols = ['推理模型', '调度算法', '批次大小']
        if is_by_type and '任务类型' in df.columns:
            sort_cols.append('任务类型')
        df.sort_values(by=sort_cols, inplace=True)
        return df.reset_index(drop=True)

    # ---------------- 1. 按 全局 分组统计 ----------------
    overall_grouped = full_df.groupby(['model_name', 'schedule_algorithm', 'batch_size_config'])
    overall_df = overall_grouped.agg(**base_agg_dict).reset_index()
    overall_df = process_precision_and_rename(overall_df, is_by_type=False)
    overall_df.to_csv(overall_output, index=False, encoding='utf-8-sig')

    # ---------------- 2. 按 任务类型 (task_type) 分组统计 ----------------
    by_type_grouped = full_df.groupby(['model_name', 'schedule_algorithm', 'batch_size_config', 'task_type'])
    by_type_df = by_type_grouped.agg(**base_agg_dict).reset_index()
    by_type_df = process_precision_and_rename(by_type_df, is_by_type=True)
    by_type_df.to_csv(by_type_output, index=False, encoding='utf-8-sig')

    print(f"\n分析完成！")
    print(f"1. 全局统计结果保存至: {overall_output}")
    print(f"2. 按任务类型 (task_type) 细分统计保存至: {by_type_output}\n")

    print("=== 按任务类型 (task_type) 统计示例 ===")
    print(by_type_df.to_string())

    return overall_df, by_type_df


if __name__ == "__main__":
    LOGS_DIRECTORY = "./for_scheduler/blendserve/"
    analyze_task_logs(
        input_dir=LOGS_DIRECTORY,
        overall_output=f"{LOGS_DIRECTORY}task_overall_summary.csv",
        by_type_output=f"{LOGS_DIRECTORY}task_by_type_summary.csv"
    )

