# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/7/31 21:51
@Description: token输出长度预测模型结果性能分析
"""
import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr  # 新增 Spearman 相关系数计算


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

    # P95 基础指标
    p95_mae = mean_absolute_error(y_p95_true, y_p95_pred)
    p95_rmse = np.sqrt(mean_squared_error(y_p95_true, y_p95_pred))
    p95_mape = calculate_mape(y_p95_true, y_p95_pred)
    p95_r2 = r2_score(y_p95_true, y_p95_pred) if count > 1 else np.nan

    # ---------------------------------------------------------
    # 新增 P95 长尾特性评估指标
    # ---------------------------------------------------------
    # 1. 覆盖率 (Coverage Rate, %): 预测值大于等于真实值的比例
    p95_coverage = np.mean(y_p95_pred >= y_p95_true) * 100

    # 2. Spearman 秩相关系数: 评估排序的一致性
    if count > 1 and np.std(y_p95_true) > 0 and np.std(y_p95_pred) > 0:
        p95_spearman, _ = spearmanr(y_p95_true, y_p95_pred)
    else:
        p95_spearman = np.nan

    return pd.Series({
        'sample_count': count,
        'Med_MAE': med_mae,
        'Med_RMSE': med_rmse,
        'Med_MAPE(%)': med_mape,
        'Med_R2': med_r2,
        'P95_MAE': p95_mae,
        'P95_RMSE': p95_rmse,
        'P95_MAPE(%)': p95_mape,
        'P95_R2': p95_r2,
        'P95_Coverage(%)': p95_coverage,  # 新增
        'P95_Spearman': p95_spearman  # 新增
    })


def aggregate_mean_std(df_list, index_names):
    """
    将多次运行的结果列表进行纵向拼接，并按照指定的层级索引计算 mean ± std
    """
    if not df_list:
        return pd.DataFrame()

    concat_df = pd.concat(df_list)

    mean_df = concat_df.groupby(level=index_names).mean()
    std_df = concat_df.groupby(level=index_names).std().fillna(0)

    result_df = pd.DataFrame(index=mean_df.index)

    for col in mean_df.columns:
        if col in ['sample_count', '样本数']:
            result_df[col] = mean_df[col].astype(int).astype(str)
        elif any(x in col for x in ['R2', 'Spearman']):
            # 相关系数和R2保留4位小数
            result_df[col] = mean_df[col].map("{:.4f}".format) + " ± " + std_df[col].map("{:.4f}".format)
        else:
            # 其他误差及百分比(Coverage/MAPE)保留2位小数
            result_df[col] = mean_df[col].map("{:.2f}".format) + " ± " + std_df[col].map("{:.2f}".format)

    return result_df


def main():
    folder_path = "./model_0731"
    file_pattern = os.path.join(folder_path, "predictions_detail_run*.csv")
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"未找到任何匹配的文件: {file_pattern}")
        return

    print("=" * 80)
    print(f"成功找到 {len(file_list)} 个评估文件，开始批量读取与合并计算...")
    print("=" * 80)

    col_names = [
        'prompt_id', 'sub_id', 'prompt_text', 'dataset_source', 'prompt_type',
        'inference_model', 'lengths_raw', 'sample_count', 'repeat_flag', 'stats_dict',
        'real_median', 'real_p95', 'pred_median', 'pred_p95',
        'median_ae', 'p95_ae', 'median_ape', 'p95_ape'
    ]

    dim1_results = []
    dim2_results = []
    dim3_results = []

    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) == len(col_names) and not isinstance(df.columns[0], str):
                df.columns = col_names
            elif len(df.columns) == len(col_names):
                df.columns = col_names
        except Exception:
            df = pd.read_csv(file_path, names=col_names, header=None)

        prompt_stats = df.groupby('prompt_type').agg(
            样本数=('prompt_id', 'count'),
            真实Median均值=('real_median', 'mean'),
            真实Median中位数=('real_median', 'median'),
            真实P95均值=('real_p95', 'mean'),
            真实P95最大值=('real_p95', 'max')
        )
        dim1_results.append(prompt_stats)

        overall_model_perf = df.groupby('inference_model', group_keys=False).apply(evaluate_group)
        dim2_results.append(overall_model_perf)

        cross_perf = df.groupby(['inference_model', 'prompt_type'], group_keys=False).apply(evaluate_group)
        dim3_results.append(cross_perf)

    print("\n【维度一】不同 Prompt 类型的真实输出长度分布特征 (Mean ± Std)：")
    final_dim1 = aggregate_mean_std(dim1_results, index_names=['prompt_type'])
    print(final_dim1.to_string())

    print("\n\n【维度二】各推理模型的整体预测误差评估 (Mean ± Std)：")
    final_dim2 = aggregate_mean_std(dim2_results, index_names=['inference_model'])
    print(final_dim2.to_string())

    print("\n\n【维度三】模型在不同类型 Prompt 上的细粒度表现对比 (Mean ± Std)：")
    final_dim3 = aggregate_mean_std(dim3_results, index_names=['inference_model', 'prompt_type'])
    print(final_dim3.to_string())

    output_report_path = os.path.join(folder_path, "summary_analysis_report_0731_p95.csv")
    final_dim3.to_csv(output_report_path)
    print(f"\n" + "=" * 80)
    print(f"聚合分析结果 (Mean ± Std) 已成功导出至文件: {output_report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
