# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：build_token_output_length_dataset.py
@Author  ：Zhihao Zhang
@Date    ：2026/7/9 10:39 
@Describe：构建Token输出长度数据集
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from object.dataset_loader import DatasetLoader, extract_prompt
from utils import save_result

# 模型名称及本地路径
MODEL_NAME = "Qwen2.5-7B"
# MODEL_NAME = "Deepseek-R1-1.5B"
# MODEL_NAME = "Llama3-8B"

MODEL_PATH = "/home/hao/Workspace/models/qwen/Qwen2.5-7B-Instruct"
# MODEL_PATH = "/home/hao/Workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_PATH = "/home/hao/Workspace/models/meta-llama/Meta-Llama-3-8B-Instruct"

# 数据集名称、路径、类型
DATASET_NAME = "allenai/WildChat"
DATASET_LOCAL_PATH = "/home/hao/Workspace/datasets/WildChat"
DATASET_TYPE = "chat"

# 数据保存路径
SAVE_DIR = "./output"
CSV_PATH = f"{SAVE_DIR}/{MODEL_NAME}/results.csv"

# 数据集取多少个Prompt
NUM_PROMPTS = 5
# 批处理大小
BATCH_SIZE = 2
# 最大输出token数量
MAX_NEW_TOKENS = 1024

# 模型参数
TEMPERATURE = 0.8
TOP_P = 0.95
DO_SAMPLE = True
DTYPE = torch.float16
DEVICE_MAP = "auto"
SEED = 42

# APS参数
MEDIAN_THRESHOLD = 0.03  # 与中位数允许偏差(3%)
MAX_REPEAT_TIMES = 16  # 每个Prompt最多采样次数
MIN_REPEAT_TIMES = 3  # 每个Prompt最少采样次数
MIN_TOKEN_DELTA = 3  # 至少允许3token

# 推理设备
if not torch.cuda.is_available():
    print("Running on CPU")
    raise Exception("CUDA not available")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 加载模型
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=DTYPE, device_map="auto", trust_remote_code=True)
model.eval()

# 加载数据集
print("Loading WildChat dataset...")
dataset = DatasetLoader(DATASET_NAME, DATASET_LOCAL_PATH, DATASET_TYPE)
samples = dataset.extract_data(SEED, NUM_PROMPTS)
prompts = extract_prompt(samples)

# 当前要运行的batch
active_batch = []
# 下一个带加入batch的prompt
next_prompt_idx = 0

# 初始化batch
while len(active_batch) < BATCH_SIZE and next_prompt_idx < len(prompts):
    active_batch.append(prompts[next_prompt_idx])
    next_prompt_idx += 1
print("Batch初始化成功，开始批处理……")

"""
主循环
只要Batch里还有Prompt，就继续推理。
"""
while len(active_batch) > 0:
    # 1. 取出prompt文本
    batch_prompts = [prompt.prompt_text for prompt in active_batch]

    # 2. tokenizer
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # 将所有Tensor移动到GPU
    inputs = {
        k: v.cuda()
        for k, v in inputs.items()
    }

    # 计算每个Prompt自己的输入长度
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    # 3. generate
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 4. 更新Prompt对象
    for idx, prompt in enumerate(active_batch):
        # 获取模型新生成的 Token
        generated_ids = outputs[idx][input_lengths[idx]:]
        # 解码生成文本
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # 输出 Token 数量
        output_length = len(generated_ids)
        # 更新 Prompt 对象
        prompt.add_output_result(output_length, output_text)
        # 判断是否已经稳定
        prompt.finished = prompt.is_stable(MIN_REPEAT_TIMES, MAX_REPEAT_TIMES, MEDIAN_THRESHOLD, MIN_TOKEN_DELTA)

    # 5. 删除已经Finished的Prompt
    new_active_batch = []
    for prompt in active_batch:
        if prompt.finished:
            # 如果该prompt稳定，先保存，再从本batch剔除
            save_result.append_prompt_to_csv(prompt, CSV_PATH)
            print(f"Prompt {prompt.prompt_id} Finished after {prompt.repeat_times} iterations.")
        else:
            new_active_batch.append(prompt)
    active_batch = new_active_batch

    # 6. 加入新的Prompt
    while len(active_batch) < BATCH_SIZE and next_prompt_idx < len(prompts):
        active_batch.append(prompts[next_prompt_idx])
        print(f"Add Prompt {prompts[next_prompt_idx].prompt_id} into Active Batch.")
        next_prompt_idx += 1
