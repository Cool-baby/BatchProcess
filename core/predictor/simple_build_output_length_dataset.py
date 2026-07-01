# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：simple_build_output_length_dataset.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/13 22:54 
@Describe：
"""
import json
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from object.dataset_loader import DatasetLoader


# 配置
MODEL_PATH = "/home/hao/Workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET_LOCAL_PATH = "/home/hao/Workspace/datasets/WildChat"
DATASET_NAME = "allenai/WildChat"
OUTPUT_FILE = "wildchat_length_dataset_051303.jsonl"
NUM_PROMPTS = 2000
K = 4
MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 1024

TEMPERATURE = 1.0
TOP_P = 0.95

DEVICE = "cuda"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

# 加载数据集
dataset = DatasetLoader(DATASET_NAME, DATASET_LOCAL_PATH,"chat")
samples = dataset.extract_data(42, NUM_PROMPTS)

# 提取prompt
prompts = []

for sample in samples:

    try:
        conv = sample["conversation"]
        if len(conv) == 0:
            continue
        if conv[0]["role"] != "user":
            continue
        prompt = conv[0]["content"].strip()
        token_num = len(tokenizer.encode(prompt))
        if token_num < 4:
            continue
        if token_num > MAX_INPUT_LENGTH:
            continue
        prompts.append(prompt)
    except:
        continue

print(f"Collected prompts: {len(prompts)}")


# 主循环
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx, prompt in enumerate(tqdm(prompts)):
        sampled_lengths = []
        finish_reasons = []
        for _ in range(K):
            # 随机seed
            seed = random.randint(0, 10**9)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_LENGTH
            ).to(DEVICE)
            input_length = inputs.input_ids.shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            output_ids = outputs[0][input_length:]
            output_ids = output_ids.tolist()

            # 正确统计真实长度
            if tokenizer.eos_token_id in output_ids:
                eos_pos = output_ids.index(
                    tokenizer.eos_token_id
                )
                real_length = eos_pos + 1
                finish_reason = "eos"
            else:
                real_length = len(output_ids)
                finish_reason = "length"

            sampled_lengths.append(real_length)
            finish_reasons.append(finish_reason)

        result = {
            "id": idx,
            "prompt": prompt,
            "prompt_length": input_length,
            "sampled_lengths": sampled_lengths,
            "finish_reasons": finish_reasons
        }

        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()

print("Done.")
