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
MODEL_INFO = {"model_name": "Gemma-3-4b-Instruct", "model_path": "/home/hao/Workspace/models/google/gemma-3-4b-it" }

# 数据集名称、路径、类型
DATASET_INFO = {"dataset_name": "WildChat", "full_name": "allenai/WildChat", "path": "/home/hao/Workspace/datasets/WildChat", "type": "chat"}

# 数据保存路径
SAVE_DIR = "output"
CSV_PATH = f"{SAVE_DIR}/{MODEL_INFO['model_name']}/results-{MODEL_INFO['model_name']}-{DATASET_INFO['dataset_name']}.csv"

# 检查目录是否存在，如果不存在，则创建它
directory = os.path.dirname(CSV_PATH)
if not os.path.exists(directory):
    os.makedirs(directory)
# 日志记录
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(f"{SAVE_DIR}/{MODEL_INFO['model_name']}/run.log", encoding="utf-8")]
)

# 数据集取多少个Prompt
NUM_PROMPTS = 1200
# 批处理大小
BATCH_SIZE = 8
# 最大输出token数量
MAX_NEW_TOKENS = 2048
# 最大输入token数量
MAX_INPUT_TOKENS = 2048

# 模型参数
TEMPERATURE = 0.8
TOP_P = 0.95
DO_SAMPLE = True
DTYPE = torch.bfloat16
DEVICE_MAP = "auto"
SEED = 42

# APS参数
MEDIAN_THRESHOLD = 0.05  # 与中位数允许偏差(5%)
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
# 避免 pad_token 与 eos_token 相同导致的早停失效
if "<pad>" in tokenizer.get_vocab():
    tokenizer.pad_token = "<pad>"
elif tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 获取 Gemma-3 对话模式下的所有结束符 (包含 <eos> 和 <end_of_turn>)
eos_token_ids = [tokenizer.eos_token_id]
end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
if end_of_turn_id is not None and end_of_turn_id != tokenizer.unk_token_id:
    eos_token_ids.append(end_of_turn_id)

# 收集所有的停止 Token ID 用于后续精确切片判定
stop_token_ids = set(eos_token_ids + [tokenizer.pad_token_id])

# 加载模型
logging.info(f"Loading model - {MODEL_INFO['model_name']}")
model = AutoModelForCausalLM.from_pretrained(MODEL_INFO["model_path"], dtype=DTYPE, device_map="auto", trust_remote_code=True)
model.eval()
logging.info(model.hf_device_map)

# 加载数据集
logging.info(f"Loading dataset - {DATASET_INFO['dataset_name']}")
dataset = DatasetLoader(DATASET_INFO['full_name'], DATASET_INFO['path'], DATASET_INFO['type'])
samples = dataset.extract_data(SEED, NUM_PROMPTS)
prompts = extract_prompt(samples, tokenizer, MAX_INPUT_TOKENS)

# 当前要运行的batch
active_batch = []
# 实际加载了多少条数据
number_of_real_data = len(prompts)
# 下一个待加入batch的prompt
next_prompt_idx = 0
# 记录已经处理了多少个prompt
prompt_finished_counter = 0
# 记录已经处理了多少个Batch
batch_finished_counter = 0

# 初始化batch
while len(active_batch) < BATCH_SIZE and next_prompt_idx < number_of_real_data:
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
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    # 将所有Tensor移动到GPU
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # 记录该 Batch 统一 Padding 后的输入序列长度
    batch_input_len = inputs["input_ids"].shape[1]

    # 3. generate (重点：传入 eos_token_id=eos_token_ids 列表)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.1,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_ids,  # 传入包含 <end_of_turn> 的列表
        )

    # 4. 更新 Prompt 对象
    for idx, prompt in enumerate(active_batch):
        # 切除 Prompt 部分，仅保留新生成的 Token
        generated_sequence = outputs[idx][batch_input_len:].tolist()

        # 寻找生成的 Token 中最先出现的终止符号（<eos> / <end_of_turn> / <pad>）
        stop_indices = [
            i for i, token_id in enumerate(generated_sequence) if token_id in stop_token_ids
        ]

        if stop_indices:
            # 遇到了停止符，截断到第一个停止符之前
            first_stop_idx = stop_indices[0]
            true_generated_ids = generated_sequence[:first_stop_idx]
        else:
            # 确实跑满了 1024 Token 且没遇到停止符
            true_generated_ids = generated_sequence

        # 解码生成文本
        output_text = tokenizer.decode(true_generated_ids, skip_special_tokens=True)
        # 真实的输出 Token 数量
        output_length = len(true_generated_ids)

        # 1. 重新对解码得到的文本进行 tokenize（不加 BOS/EOS 等特殊符号）
        re_encoded_ids = tokenizer.encode(output_text, add_special_tokens=False)
        re_encoded_len = len(re_encoded_ids)

        # 2. 打印对比日志
        if abs(output_length - re_encoded_len) > 3:
            logging.warning(
                f"[警告] Prompt {prompt.prompt_id} 长度不匹配！\n"
                f"  - 切片得到的 Token 数: {output_length}\n"
                f"  - 二次 Encode Token 数: {re_encoded_len}\n"
                f"  - 解码文本前 30 字: {repr(output_text[:30])}\n"
                f"  - 原始切片前 5 个 ID: {true_generated_ids[:5]}\n"
                f"  - 原始切片后 5 个 ID: {true_generated_ids[-5:]}"
            )

        first_5_words = tokenizer.decode(true_generated_ids[:5])
        last_5_words = tokenizer.decode(true_generated_ids[-5:])
        print(f"回答开头: {repr(first_5_words)} | 回答结尾: {repr(last_5_words)}")

        # 更新 Prompt 对象
        prompt.add_output_result(output_length, MODEL_INFO['model_name'], output_text)
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
            logging.info(f"Processed: {prompt_finished_counter}, Remaining: {number_of_real_data - prompt_finished_counter}, Progress: {(prompt_finished_counter / number_of_real_data) * 100:.2f}%")
    active_batch = new_active_batch

    # 6. 加入新的Prompt
    while len(active_batch) < BATCH_SIZE and next_prompt_idx < len(prompts):
        active_batch.append(prompts[next_prompt_idx])
        # logging.info(f"Add Prompt {prompts[next_prompt_idx].prompt_id} into Active Batch.")
        next_prompt_idx += 1

    batch_finished_counter += 1

    # 7. 显存清理，防止 OOM
    del inputs
    del outputs
    # 释放没用的显存碎片，每跑 5-10 个 Batch 清理一次
    if batch_finished_counter % 10 == 0:
        torch.cuda.empty_cache()

    logging.info(f"Batch Processed: {batch_finished_counter}")

logging.info("Finished processing.")
