# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：build_output_length_dataset.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/12 11:00 
@Describe：构建训练输出长度预测器的数据集
"""
import json
import os
import time
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import random
from object.dataset_loader import DatasetLoader


# 模型路经
MODEL_PATH = "/home/hao/Workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 数据集路径
DATASET_LOCAL_PATH = "/home/hao/Workspace/datasets/WildChat"
# 数据集名称、路径、类型、种子
DATASET_NAME = "allenai/WildChat"
DATASET_TYPE = "chat"
DATASET_SELECT_SEED = 42
# 数据集输出路径
OUTPUT_FILE = "../../data/predict_dataset/wildchat_outputlegth_dataset_051302.jsonl"
# embedding
EMBEDDING_DIR = "../../data/predict_dataset/embeddings"

# 最大Prompt数量
NUM_PROMPTS = 50
# 单Prompt统计次数（K太小无法看到 heavy-tail，太大生成成本爆炸）
K = 16
# batch大小（一次并行多少个不同prompt）
BATCH_SIZE = 4
# 输入最大长度
MAX_INPUT_LENGTH = 1024
# 最大输出Token（max_new_tokens如果太小heavy-tail容易被截断，如果太大推理极慢）
MAX_NEW_TOKENS = 1024
# 调整生成的随机性，控制生成风格的严谨或随意
TEMPERATURE = 0.8
# 控制候选词累积概率，确保合理的候选词动态加入
TOP_P = 0.95
# 限制候选词数量，使生成内容集中。
TOP_K = 50
# prompt长度限制
MIN_PROMPT_TOKENS = 8
MAX_PROMPT_TOKENS = 1024
# embedding提取层
EMBEDDING_LAYER = -1

# 推理设备
if not torch.cuda.is_available():
    print("Running on CPU")
    raise Exception("CUDA not available")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUDA优化
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 加载tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

# 加载模型
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()
print("Model loaded.")

# 加载 WildChat 数据集
print("Loading WildChat dataset...")
dataset = DatasetLoader(DATASET_NAME, DATASET_LOCAL_PATH, DATASET_TYPE)
samples = dataset.extract_data(DATASET_SELECT_SEED, NUM_PROMPTS)
print(f"Loaded samples:\n{samples}")

# 提取用户prompt
prompts = []

for sample in samples:
    try:
        conversations = sample["conversation"]
        if len(conversations) == 0:
            continue
        first_msg = conversations[0]
        if first_msg["role"] != "user":
            continue
        prompt = first_msg["content"].strip()
        # 过滤太短prompt
        if len(prompt) < 10:
            continue

        # token长度过滤
        token_num = len(tokenizer.encode(prompt))
        if token_num < MIN_PROMPT_TOKENS:
            continue
        if token_num > MAX_PROMPT_TOKENS:
            continue
        prompts.append(prompt)

    except Exception:
        continue

print(f"Collected {len(prompts)} valid prompts.")

# 创建embedding目录
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# 断点恢复
processed = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        processed = sum(1 for _ in f)
print(f"Already processed: {processed}")


# 主循环
output_file = open(OUTPUT_FILE, "a", encoding="utf-8")

# 按prompt组织结果
all_prompt_results = {}

# 初始化已处理prompt
for idx in range(processed):
    all_prompt_results[idx] = None

# batch循环
for batch_start in tqdm(range(processed, len(prompts), BATCH_SIZE)):

    batch_end = min(batch_start + BATCH_SIZE, len(prompts))
    batch_prompts = prompts[batch_start:batch_end]
    current_batch_size = len(batch_prompts)

    # 初始化batch结果
    batch_results = {}

    for local_idx in range(current_batch_size):
        print(f"[{time.time()}] -> for local_idx in range(current_batch_size)")
        global_idx = batch_start + local_idx
        batch_results[global_idx] = {
            "id": global_idx,
            "prompt": batch_prompts[local_idx],
            "embedding_path": f"{EMBEDDING_DIR}/{global_idx}.npy",
            "prompt_length": None,
            "sampled_lengths": [],
            "finish_reasons": [],
            "generation_times": [],
            "generated_previews": [],
            "censored_count": 0
        }

    # embedding提取（每个prompt只一次）
    embedding_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).to(DEVICE)

    with torch.no_grad():
        embedding_outputs = model(input_ids=embedding_inputs.input_ids, attention_mask=embedding_inputs.attention_mask, output_hidden_states=True, return_dict=True)

    hidden_states = embedding_outputs.hidden_states[-1]

    for local_idx in range(current_batch_size):
        print(f"[{time.time()}] -> for local_idx in range(current_batch_size)")
        global_idx = batch_start + local_idx
        hidden = hidden_states[
            local_idx,
            -1
        ].float().cpu().numpy()
        embedding_path = batch_results[
            global_idx
        ]["embedding_path"]
        if not os.path.exists(embedding_path):
            np.save(embedding_path, hidden)

    del embedding_outputs
    del hidden_states
    torch.cuda.empty_cache()

    # K次独立采样
    for k in range(K):
        print(f"[{time.time()}] -> for k in range(K)")
        try:
            # 不同随机seed
            seed = random.randint(0, 10**9)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # tokenize
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).to(DEVICE)

            input_lengths = (inputs.attention_mask.sum(dim=1).tolist())

            # CUDA同步计时
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False
                )

            torch.cuda.synchronize()
            end_time = time.time()
            generation_time = end_time - start_time

            sequences = outputs.sequences

            # 保存每个prompt结果
            for local_idx in range(current_batch_size):

                global_idx = batch_start + local_idx
                input_length = int(input_lengths[local_idx])
                output_ids = sequences[local_idx]
                generated_ids = output_ids[input_length:]

                output_length = len(generated_ids)

                batch_results[global_idx]["sampled_lengths"].append(int(output_length))

                batch_results[global_idx]["generation_times"].append(float(generation_time))

                # EOS判断
                if len(generated_ids) > 0 and generated_ids[-1] == tokenizer.eos_token_id:
                    finish_reason = "eos"
                else:
                    finish_reason = "length"
                    batch_results[global_idx]["censored_count"] += 1

                batch_results[
                    global_idx
                ]["finish_reasons"].append(
                    finish_reason
                )

                # preview
                generated_text = tokenizer.decode(
                    generated_ids[:128],
                    skip_special_tokens=True
                )

                batch_results[
                    global_idx
                ]["generated_previews"].append(
                    generated_text
                )

                # 保存prompt长度
                if (
                    batch_results[global_idx]
                    ["prompt_length"] is None
                ):
                    batch_results[
                        global_idx
                    ]["prompt_length"] = (
                        input_length
                    )

            del outputs
            del sequences
            del inputs

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nGeneration Error at batch {batch_start}, sample {k}: {e}")
            continue

    # batch结束后统一写入
    for global_idx in sorted(batch_results.keys()):
        print(f"[{time.time()}] -> for global_idx in sorted(batch_results.keys()):")
        result = batch_results[global_idx]
        sampled_lengths = result[
            "sampled_lengths"
        ]

        if len(sampled_lengths) == 0:
            continue

        result["mean_length"] = float(
            np.mean(sampled_lengths)
        )

        result["median_length"] = float(
            np.median(sampled_lengths)
        )

        result["p90_length"] = float(
            np.percentile(sampled_lengths, 90)
        )

        result["p95_length"] = float(
            np.percentile(sampled_lengths, 95)
        )

        result["p99_length"] = float(
            np.percentile(sampled_lengths, 99)
        )

        result["max_length"] = int(
            np.max(sampled_lengths)
        )

        result["min_length"] = int(
            np.min(sampled_lengths)
        )

        result["std_length"] = float(
            np.std(sampled_lengths)
        )

        result["all_eos"] = (
            result["censored_count"] == 0
        )

        result["sampling_config"] = {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS
        }

        output_file.write(
            json.dumps(
                result,
                ensure_ascii=False
            ) + "\n"
        )

        output_file.flush()

output_file.close()

print("Dataset collection completed.")
