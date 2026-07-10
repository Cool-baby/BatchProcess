# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：build_token_output_length_dataset.py
@Author  ：Zhihao Zhang
@Date    ：2026/7/9 10:39 
@Describe：构建Token输出长度数据集
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from object.dataset_loader import DatasetLoader, extract_prompt
from utils import save_result
import logging

# 模型名称及本地路径
# MODEL_INFO = {"model_name": "Qwen2.5-3B-Instruct", "model_path": "/home/hao/Workspace/models/qwen/Qwen2.5-3B-Instruct" }
MODEL_INFO = {"model_name": "Llama-3.2-3B-Instruct", "model_path": "/home/hao/Workspace/models/meta-llama/Llama-3.2-3B-Instruct" }
# MODEL_INFO = {"model_name": "Deepseek-R1-1.5B", "model_path": "/home/hao/Workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" }
# MODEL_INFO = {"model_name": "Qwen2.5-7B", "model_path": "/home/hao/Workspace/models/qwen/Qwen2.5-7B-Instruct" }
# MODEL_INFO = {"model_name": "Llama3-8B", "model_path": "/home/hao/Workspace/models/meta-llama/Meta-Llama-3-8B-Instruct" }

# 数据集名称、路径、类型
DATASET_NAME = "allenai/WildChat"
DATASET_LOCAL_PATH = "/home/hao/Workspace/datasets/WildChat"
DATASET_TYPE = "chat"

# 数据保存路径
SAVE_DIR = "./output"
CSV_PATH = f"{SAVE_DIR}/{MODEL_INFO["model_name"]}/results.csv"

# 检查目录是否存在，如果不存在，则创建它
directory = os.path.dirname(CSV_PATH)
if not os.path.exists(directory):
    os.makedirs(directory)
# 日志记录
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"{SAVE_DIR}/{MODEL_INFO["model_name"]}/run.log", encoding="utf-8")]
)

# 数据集取多少个Prompt
NUM_PROMPTS = 100
# 批处理大小
BATCH_SIZE = 4
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
MEDIAN_THRESHOLD = 0.05  # 与中位数允许偏差(3%)
MAX_REPEAT_TIMES = 16  # 每个Prompt最多采样次数
MIN_REPEAT_TIMES = 3  # 每个Prompt最少采样次数
MIN_TOKEN_DELTA = 3  # 至少允许3token

# 推理设备
if not torch.cuda.is_available():
    logging.info("No GPU found, using CPU")
    raise Exception("CUDA not available")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载tokenizer
logging.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_INFO["model_path"], trust_remote_code=True)
if "Llama" in MODEL_INFO["model_name"]:  # 如果是Llama模型，需要手动设置
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_NEW_TOKENS

# 加载模型
logging.info(f"Loading model - {MODEL_INFO['model_name']}")
model = AutoModelForCausalLM.from_pretrained(MODEL_INFO["model_path"], dtype=DTYPE, device_map="auto", trust_remote_code=True)
model.eval()
logging.info(model.hf_device_map)

# 加载数据集
logging.info(f"Loading dataset - {DATASET_NAME}")
dataset = DatasetLoader(DATASET_NAME, DATASET_LOCAL_PATH, DATASET_TYPE)
samples = dataset.extract_data(SEED, NUM_PROMPTS)
prompts = extract_prompt(samples)

# 当前要运行的batch
active_batch = []
# 下一个待加入batch的prompt
next_prompt_idx = 0
# 记录已经处理了多少个prompt
prompt_finished_counter = 0
# 记录已经处理了多少个Batch
batch_finished_counter = 0

# 初始化batch
while len(active_batch) < BATCH_SIZE and next_prompt_idx < len(prompts):
    active_batch.append(prompts[next_prompt_idx])
    next_prompt_idx += 1
logging.info("Batch initialization successful, starting batch processing...")

"""
主循环, 只要Batch里还有Prompt，就继续推理。
"""
while len(active_batch) > 0:
    # 1. 取出prompt文本
    batch_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in active_batch
    ]

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
            repetition_penalty=1.1,
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
        # logging.info(f"{prompt.prompt_text}: <{output_length}>\n{output_text}")
        # print(f"input_length: {input_lengths[idx]}, output_total_length: {len(outputs[idx])}, generated_length: {len(generated_ids)}, decoded: {tokenizer.decode(generated_ids, skip_special_tokens=True)}")
        # 更新 Prompt 对象
        prompt.add_output_result(output_length, output_text)
        # 判断是否已经稳定
        prompt.finished = prompt.is_stable(MIN_REPEAT_TIMES, MAX_REPEAT_TIMES, MEDIAN_THRESHOLD, MIN_TOKEN_DELTA)

    # 5. 删除已经Finished的Prompt
    new_active_batch = []
    for prompt in active_batch:
        if prompt.finished == -1:
            new_active_batch.append(prompt)
        else:
            # 如果该prompt稳定，先保存，再从本batch剔除
            save_result.append_prompt_to_csv(prompt, CSV_PATH)
            logging.info(f"Prompt {prompt.prompt_id} Finished after {prompt.repeat_times} iterations.")
            prompt_finished_counter += 1
            logging.info(f"Processed: {prompt_finished_counter}, Remaining: {NUM_PROMPTS - prompt_finished_counter}, Progress: {(prompt_finished_counter / NUM_PROMPTS) * 100:.2f}%")
    active_batch = new_active_batch

    # 6. 加入新的Prompt
    while len(active_batch) < BATCH_SIZE and next_prompt_idx < len(prompts):
        active_batch.append(prompts[next_prompt_idx])
        # logging.info(f"Add Prompt {prompts[next_prompt_idx].prompt_id} into Active Batch.")
        next_prompt_idx += 1

    batch_finished_counter += 1
    logging.info(f"Batch Processed: {batch_finished_counter}")

logging.info("Finished processing.")
