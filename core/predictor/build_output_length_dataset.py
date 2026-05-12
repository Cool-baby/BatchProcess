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
from transformers import AutoModel
from object.dataset_loader import DatasetLoader


# 模型路经
MODEL_PATH = "/home/hao/Workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 数据集名称、路径、类型、种子
DATASET_NAME = "allenai/WildChat"
DATASET_LOCAL_PATH = "/home/hao/Workspace/datasets/WildChat"
DATASET_TYPE = "chat"
DATASET_SELECT_SEED = 42
# 数据集输出路径
OUTPUT_FILE = "../../data/predict_dataset/wildchat_outputlegth_dataset_051202.jsonl"
# embedding
EMBEDDING_DIR = "../../data/predict_dataset/embeddings"
# 推理设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 最大Prompt数量
NUM_PROMPTS = 50
# 单Prompt统计次数（K太小无法看到 heavy-tail，太大生成成本爆炸）
K = 16
# 最大输出Token（max_new_tokens如果太小heavy-tail容易被截断，如果太大推理极慢）
MAX_NEW_TOKENS = 2048
# 调整生成的随机性，控制生成风格的严谨或随意
TEMPERATURE = 0.8
# 控制候选词累积概率，确保合理的候选词动态加入
TOP_P = 0.95
# 限制候选词数量，使生成内容集中。
TOP_K = 50

# 加载tokenizer和model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="auto", trust_remote_code=True)
# 用于hidden state提取
embedding_model = AutoModel.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()
embedding_model.eval()
print("Model loaded.")

# 加载 WildChat
print("Loading WildChat dataset...")
dataset = DatasetLoader(DATASET_NAME, DATASET_LOCAL_PATH, DATASET_TYPE)
samples = dataset.extract_data(DATASET_SELECT_SEED, NUM_PROMPTS)
print(f"Loaded samples:\n{samples}")

# 提取用户prompt
prompts = []
for sample in samples:
    try:
        # WildChat message格式
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
        if token_num < 8:
            continue
        if token_num > 1024:
            continue
        prompts.append(prompt)
    except:
        continue

print(f"Collected {len(prompts)} prompts.")


# 断点恢复
processed = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        processed = sum(1 for _ in f)
print(f"Already processed: {processed}")


# 主循环
output_file = open(OUTPUT_FILE, "a")
for idx in tqdm(range(processed, len(prompts))):

    prompt = prompts[idx]
    try:
        # tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
        input_length = inputs.input_ids.shape[1]

        # 提取并保存 hidden embedding
        embedding_path = f"{EMBEDDING_DIR}/{idx}.npy"
        if not os.path.exists(embedding_path):
            with torch.no_grad():
                embedding_outputs = embedding_model(**inputs, output_hidden_states=True)

            hidden = embedding_outputs.hidden_states[-1][:, -1, :]
            hidden = hidden.squeeze(0).float().cpu().numpy()

            np.save(embedding_path, hidden)

        # K次采样
        sampled_lengths = []
        finish_reasons = []
        generation_times = []
        generated_previews = []
        censored_count = 0

        for k in range(K):
            # 不固定generation seed
            seed = int(time.time() * 1000000) % 100000000
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

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
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )

            torch.cuda.synchronize()
            end_time = time.time()
            generation_time = end_time - start_time

            # 输出token
            output_ids = outputs.sequences[0]
            generated_ids = output_ids[input_length:]
            output_length = len(generated_ids)
            sampled_lengths.append(int(output_length))
            generation_times.append(float(generation_time))

            # EOS判断
            if generated_ids[-1] == tokenizer.eos_token_id:
                finish_reason = "eos"
            else:
                finish_reason = "length"
                censored_count += 1

            finish_reasons.append(finish_reason)

            # 保存部分生成文本（方便后续分析）
            generated_text = tokenizer.decode(
                generated_ids[:128],
                skip_special_tokens=True
            )

            generated_previews.append(generated_text)

        # 是否全部EOS
        all_eos = (censored_count == 0)

        # 统计量
        result = {
            "id": idx,
            "prompt": prompt,
            "embedding_path": embedding_path,
            "prompt_length": int(input_length),
            "sampled_lengths": sampled_lengths,
            "mean_length": float(np.mean(sampled_lengths)),
            "median_length": float(np.median(sampled_lengths)),
            "p90_length": float(np.percentile(sampled_lengths, 90)),
            "p95_length": float(np.percentile(sampled_lengths, 95)),
            "p99_length": float(np.percentile(sampled_lengths, 99)),
            "max_length": int(np.max(sampled_lengths)),
            "min_length": int(np.min(sampled_lengths)),
            "std_length": float(np.std(sampled_lengths)),
            "finish_reasons": finish_reasons,
            "generation_times": generation_times,
            "generated_previews": generated_previews,
            "censored_count": censored_count,
            "all_eos": all_eos,
            "sampling_config": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "max_new_tokens": MAX_NEW_TOKENS
            }
        }

        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        output_file.flush()

    except Exception as e:
        print(f"\nError at sample {idx}: {e}")
        continue

output_file.close()

print("Dataset collection completed.")
