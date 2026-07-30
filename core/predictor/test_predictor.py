# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/7/29 20:30
@Description: token输出长度预测器
"""
import os
import time
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from core.predictor.token_length_predictor import TokenLengthPredictor

# ==========================================
# 1. 日志与输出目录配置
# ==========================================
OUTPUT_DIR = "./evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(OUTPUT_DIR, f"eval_{timestamp}.log")

# 配置全局 Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# 2. 健壮的数据加载函数
# ==========================================
def load_dataset_from_jsonl(filepath: str) -> pd.DataFrame:
    """安全读取 JSONL 文件，容忍多行换行与格式异常"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据集文件不存在: {filepath}")

    records = []
    skipped_count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError as e:
                skipped_count += 1
                logger.warning(f"第 {idx} 行 JSON 解析失败已跳过，错误: {e}")

    logger.info(f"成功读取 {len(records)} 条有效数据 (跳过异常行: {skipped_count})")

    df = pd.DataFrame(records)

    # 字段存在性校验
    required_cols = ['prompt_text', 'inference_model', 'statistics']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"数据集缺少必要的字段: {missing_cols}")

    return df


# ==========================================
# 3. 评估指标计算辅助函数
# ==========================================
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    """计算单个维度的 MAE, RMSE, MAPE 和 R2Score"""
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mape = np.mean(np.abs(y_pred - y_true) / (y_true + 1e-5)) * 100

    # 防止分母为 0 计算 R2
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-5))

    return {
        f"{prefix}_MAE": round(mae, 2),
        f"{prefix}_RMSE": round(rmse, 2),
        f"{prefix}_MAPE(%)": round(mape, 2),
        f"{prefix}_R2": round(r2, 4)
    }


# ==========================================
# 4. 主流程
# ==========================================
def main():
    # 路径配置
    LOCAL_MODEL_PATH = "/Users/hao/Workspace/Other/models/Qwen2.5-1.5B"
    # LOCAL_MODEL_PATH = "/HOME/scw6e0x/run/zzhao/models/Qwen2.5-1.5B"
    CHECKPOINT_PATH = "./final_model_0729.pt"
    TEST_JSONL_PATH = "../../data/MoLD_shuffled_test.jsonl"

    logger.info("================ 程序启动 ================")
    logger.info(f"基础模型路径: {LOCAL_MODEL_PATH}")
    logger.info(f"模型权重路径: {CHECKPOINT_PATH}")
    logger.info(f"测试集路径  : {TEST_JSONL_PATH}")

    # 1. 初始化预测器
    logger.info("Step 1/6: 正在初始化预测模型...")
    start_time = time.time()
    try:
        predictor = TokenLengthPredictor(
            base_model_path=LOCAL_MODEL_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            max_length=2048
        )
        logger.info(f"模型加载完成，耗时: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        logger.error(f"模型初始化失败!", exc_info=True)
        return

    # 2. 加载数据集
    logger.info("Step 2/6: 正在解析测试数据集...")
    df_test = load_dataset_from_jsonl(TEST_JSONL_PATH)

    # 提取真实值
    df_test['real_median'] = df_test['statistics'].apply(
        lambda x: x.get('median_length', 0) if isinstance(x, dict) else 0)
    df_test['real_p95'] = df_test['statistics'].apply(lambda x: x.get('p95_length', 0) if isinstance(x, dict) else 0)

    # 3. 组装输入格式
    logger.info("Step 3/6: 构造推理 Payload...")
    eval_items = [
        {
            "prompt_text": row["prompt_text"],
            "inference_model": row["inference_model"]
        }
        for _, row in df_test.iterrows()
    ]

    # 4. 执行模型批量预测
    BATCH_SIZE = 16
    logger.info(f"Step 4/6: 开始模型批量推理 (Batch Size: {BATCH_SIZE})...")
    infer_start_time = time.time()

    try:
        predictions = predictor.predict_batch(eval_items, batch_size=BATCH_SIZE)
    except Exception as e:
        logger.error("批量预测出现未捕获的异常!", exc_info=True)
        return

    infer_cost = time.time() - infer_start_time
    qps = len(df_test) / infer_cost
    logger.info(f"推理完成！数据总量: {len(df_test)} 条, 总耗时: {infer_cost:.2f} 秒, 吞吐量: {qps:.2f} 条/秒")

    # 5. 写回预测值并计算误差
    logger.info("Step 5/6: 整理数据与误差指标...")
    df_test['pred_median'] = [p['median_length'] for p in predictions]
    df_test['pred_p95'] = [p['p95_length'] for p in predictions]

    # 详细点对点误差计算 (方便分析 Bad Case)
    df_test['abs_err_median'] = np.abs(df_test['pred_median'] - df_test['real_median'])
    df_test['abs_err_p95'] = np.abs(df_test['pred_p95'] - df_test['real_p95'])
    df_test['ape_median_%'] = (df_test['abs_err_median'] / (df_test['real_median'] + 1e-5)) * 100
    df_test['ape_p95_%'] = (df_test['abs_err_p95'] / (df_test['real_p95'] + 1e-5)) * 100

    # 6. 计算总体指标并保存报告
    logger.info("Step 6/6: 生成评估报告并持久化数据...")

    metrics_median = calculate_metrics(df_test['real_median'].values, df_test['pred_median'].values, "Median")
    metrics_p95 = calculate_metrics(df_test['real_p95'].values, df_test['pred_p95'].values, "P95")

    total_report = {**metrics_median, **metrics_p95}

    # 控制台打印评估结果
    logger.info("\n" + "=" * 20 + " 总体评估报告 " + "=" * 20)
    for k, v in total_report.items():
        logger.info(f"{k:<15}: {v}")
    logger.info("=" * 54)

    # 按 inference_model 分类汇总指标
    model_group_reports = []
    if 'inference_model' in df_test.columns:
        for model_name, group in df_test.groupby('inference_model'):
            m_med = calculate_metrics(group['real_median'].values, group['pred_median'].values, "Median")
            m_p95 = calculate_metrics(group['real_p95'].values, group['pred_p95'].values, "P95")
            model_report = {"inference_model": model_name, "sample_count": len(group), **m_med, **m_p95}
            model_group_reports.append(model_report)

        df_model_report = pd.DataFrame(model_group_reports)
        logger.info("\n" + "=" * 20 + " 分模型评估报告 " + "=" * 20)
        logger.info("\n" + df_model_report.to_string(index=False))

    # 7. 数据持久化导出
    # A. 导出现场全量字段明细 (CSV)
    details_path = os.path.join(OUTPUT_DIR, f"predictions_detail_{timestamp}.csv")
    df_test.to_csv(details_path, index=False, encoding="utf-8-sig")
    logger.info(f"已导出详细预测结果 (含 Error 明细): {details_path}")

    # B. 保存指标报告 (JSON)
    report_path = os.path.join(OUTPUT_DIR, f"metrics_report_{timestamp}.json")
    full_json_report = {
        "timestamp": timestamp,
        "dataset_size": len(df_test),
        "infer_cost_seconds": round(infer_cost, 2),
        "qps": round(qps, 2),
        "overall_metrics": total_report,
        "per_model_metrics": model_group_reports
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_json_report, f, indent=2, ensure_ascii=False)
    logger.info(f"已导出结构化评估报告: {report_path}")

    logger.info("================ 所有任务已顺利完成 ================\n")


if __name__ == "__main__":
    main()