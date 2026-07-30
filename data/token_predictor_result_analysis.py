# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/7/30 13:08
@Description: token预测器输出结果性能分析
"""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred, eps=1e-8):
    """计算 MAPE (%), 避免分母为 0"""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, eps))) * 100

def evaluate_group(group_df):
    """对传入的 DataFrame 分组计算 Median 和 P95 的核心指标"""
    count = len(group_df)
    if count == 0:
        return None

    # 取出真实值与预测值
    y_med_true = group_df['real_median'].values
    y_med_pred = group_df['pred_median'].values
    y_p95_true = group_df['real_p95'].values
    y_p95_pred = group_df['pred_p95'].values

    # Median 指标
    med_mae = mean_absolute_error(y_med_true, y_med_pred)
    med_rmse = np.sqrt(mean_squared_error(y_med_true, y_med_pred))
    med_mape = calculate_mape(y_med_true, y_med_pred)
    med_r2 = r2_score(y_med_true, y_med_pred) if count > 1 else np.nan

    # P95 指标
    p95_mae = mean_absolute_error(y_p95_true, y_p95_pred)
    p95_rmse = np.sqrt(mean_squared_error(y_p95_true, y_p95_pred))
    p95_mape = calculate_mape(y_p95_true, y_p95_pred)
    p95_r2 = r2_score(y_p95_true, y_p95_pred) if count > 1 else np.nan

    return pd.Series({
        'sample_count': count,
        'Med_MAE': round(med_mae, 2),
        'Med_RMSE': round(med_rmse, 2),
        'Med_MAPE(%)': round(med_mape, 2),
        'Med_R2': round(med_r2, 4),
        'P95_MAE': round(p95_mae, 2),
        'P95_RMSE': round(p95_rmse, 2),
        'P95_MAPE(%)': round(p95_mape, 2),
        'P95_R2': round(p95_r2, 4),
    })

def main():
    file_path = "./evaluation_results/predictions_detail_20260730_105101.csv"
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # 定义列名（针对无表头或包含固定列的 CSV 自动解析）
    col_names = [
        'prompt_id', 'sub_id', 'prompt_text', 'dataset_source', 'prompt_type',
        'inference_model', 'lengths_raw', 'sample_count', 'repeat_flag', 'stats_dict',
        'real_median', 'real_p95', 'pred_median', 'pred_p95',
        'median_ae', 'p95_ae', 'median_ape', 'p95_ape'
    ]

    try:
        # 先尝试按有表头读取，如果列数一致则匹配
        df = pd.read_csv(file_path)
        if len(df.columns) == len(col_names) and not isinstance(df.columns[0], str):
            df.columns = col_names
        elif len(df.columns) == len(col_names):
            # 强制覆盖规范化的列名，便于后续统一引用
            df.columns = col_names
    except Exception:
        df = pd.read_csv(file_path, names=col_names, header=None)

    print("=" * 80)
    print(f"成功读取数据: {file_path} | 总记录数: {len(df)} 条")
    print("=" * 80)

    # ----------------------------------------------------------------------
    # 维度一：不同类型 Prompt 的基础分布统计 (Prompt Type Profiling)
    # ----------------------------------------------------------------------
    print("\n【维度一】不同 Prompt 类型的真实输出长度分布特征：")
    prompt_stats = df.groupby('prompt_type').agg(
        样本数=('prompt_id', 'count'),
        真实Median均值=('real_median', 'mean'),
        真实Median中位数=('real_median', 'median'),
        真实P95均值=('real_p95', 'mean'),
        真实P95最大值=('real_p95', 'max')
    ).round(1)
    print(prompt_stats.to_string())

    # ----------------------------------------------------------------------
    # 维度二：不同模型整体表现 (Overall Model Performance)
    # ----------------------------------------------------------------------
    print("\n\n【维度二】各推理模型的整体预测误差评估：")
    overall_model_perf = df.groupby('inference_model', group_keys=False).apply(evaluate_group)
    print(overall_model_perf.to_string())

    # ----------------------------------------------------------------------
    # 维度三：模型 × Prompt 类型的交叉分析 (Model × Prompt Type Cross Analysis)
    # ----------------------------------------------------------------------
    print("\n\n【维度三】模型在不同类型 Prompt 上的细粒度表现对比：")
    cross_perf = df.groupby(['inference_model', 'prompt_type'], group_keys=False).apply(evaluate_group)
    print(cross_perf.to_string())

    # ----------------------------------------------------------------------
    # 导出汇总数据报告
    # ----------------------------------------------------------------------
    output_report_path = "summary_analysis_report.csv"
    cross_perf.to_csv(output_report_path)
    print(f"\n" + "=" * 80)
    print(f"细粒度分析结果已成功导出至文件: {output_report_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
