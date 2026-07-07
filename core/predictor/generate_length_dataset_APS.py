# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：generate_length_dataset_APS.py
@Author  ：Zhihao Zhang
@Date    ：2026/7/7 09:59 
@Describe： Token生成长度数据集生成器，基于Adaptive Prompt-conditioned Sampling (APS)改进
"""
import os, time, json, math, random
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- 配置参数 ---
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 可替换为本地路径
BATCH_SIZE = 128  # 每次从WildChat抽取的Prompt数量
MIN_RUNS, MAX_RUNS = 3, 10  # 每个Prompt的最小和最大采样次数
APS_THRESHOLD = 0.02  # APS中止阈值（中位数相对变化量）
OUTPUT_FILE = "wildchat_aps_dataset.jsonl"

# --- 1. 流式加载数据集 (避免内存溢出) ---
print("Loading WildChat dataset in stream mode...");
dataset = load_dataset("allenai/WildChat", streaming=True, split="train")
# 随机跳过一部分以实现“抽取”的随机性
dataset_iterator = iter(dataset.skip(random.randint(0, 1000)))

# --- 2. 初始化高效推理引擎 (榨干4090性能) ---
print("Initializing vLLM Engine...")
llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.85)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024, skip_special_tokens=True)


# --- 3. 核心流式处理与 Adaptive Prompt-conditioned Sampling (APS) ---
def get_next_batch(iterator, size):
    batch = []
    for item in iterator:
        # 提取用户的第一条对话作为Prompt
        if item.get("conversation") and len(item["conversation"]) > 0:
            batch.append(item["conversation"][0]["content"])
            if len(batch) == size: break
    return batch


current_batch = get_next_batch(dataset_iterator, BATCH_SIZE)
print(f"Loaded {len(current_batch)} prompts. Starting inference...")

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    # 跟踪每个Prompt当前的运行状态: {idx: [lengths_list]}
    active_prompts = {i: {"prompt": p, "lengths": [], "results": []} for i, p in enumerate(current_batch)}

    while active_prompts:
        # 找出本轮仍需继续采样的 Prompts
        pending_indices = list(active_prompts.keys())
        prompts_to_run = [active_prompts[idx]["prompt"] for idx in pending_indices]

        # 使用vLLM批量推理整个Batch
        start_time = time.time()
        outputs = llm.generate(prompts_to_run, sampling_params, use_tqdm=False)
        end_time = time.time();
        avg_time_per_prompt = (end_time - start_time) / len(prompts_to_run)

        finished_indices = []
        for idx, output in zip(pending_indices, outputs):
            text = output.outputs[0].text
            tokens_len = len(output.outputs[0].token_ids)

            # 记录当前运行数据
            active_prompts[idx]["lengths"].append(tokens_len)
            active_prompts[idx]["results"].append(
                {"output": text, "length": tokens_len, "latency": avg_time_per_prompt})

            # --- APS 逻辑检查 ---
            lengths = active_prompts[idx]["lengths"]
            current_runs = len(lengths)

            if current_runs >= MIN_RUNS:
                # 计算引入新样本前后的中位数变化
                prev_median = np.median(lengths[:-1]) if len(lengths) > 1 else lengths[0]
                curr_median = np.median(lengths)
                median_change = abs(curr_median - prev_median) / (prev_median + 1e-5)

                # 满足APS终止条件 或 达到最大运行次数
                if median_change < APS_THRESHOLD or current_runs >= MAX_RUNS:
                    # 计算未来必然会用到的丰富统计量
                    stats = {
                        "prompt": active_prompts[idx]["prompt"],
                        "total_runs": current_runs,
                        "mean_length": float(np.mean(lengths)),
                        "median_length": float(curr_median),
                        "std_length": float(np.std(lengths)),
                        "min_length": int(np.min(lengths)),
                        "max_length": int(np.max(lengths)),
                        "avg_latency_ms": float(np.mean([r["latency"] for r in active_prompts[idx]["results"]])) * 1000,
                        "runs_data": active_prompts[idx]["results"]
                    }
                    # 4. 即时保存，防止丢数据
                    f.write(json.dumps(stats, ensure_ascii=False) + "\n");
                    f.flush()
                    finished_indices.append(idx)

        # 从活跃字典中移除已收敛的Prompt
        for idx in finished_indices: del active_prompts[idx]
        print(f"Remaining active prompts in current batch: {len(active_prompts)}")

print(f"Batch generation completed. Results saved to {OUTPUT_FILE}")

