# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：generate_length_dataset.py
@Author  ：Zhihao Zhang
@Date    ：2026/7/1 11:23 
@Describe：Token输出长度数据集构建，Batch化推理充分利用显卡
"""
import os
import time
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from object.dataset_loader import DatasetLoader

# ====================================================
# Config
# ====================================================
# 模型路经
MODEL_PATH = "/home/hao/Workspace/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 数据集名称、路径、类型
DATASET_NAME = "allenai/WildChat"
DATASET_LOCAL_PATH = "/home/hao/Workspace/datasets/WildChat"
DATASET_TYPE = "chat"

NUM_PROMPTS = 1000
K = 10

BATCH_SIZE = 64

MAX_INPUT_LENGTH = 2048
MAX_NEW_TOKENS = 1024

TEMPERATURE = 0.8
TOP_P = 0.95

SEED = 42

OUTPUT_DIR = "length_dataset"
CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "records.csv"
)

EMBEDDING_DIR = os.path.join(
    OUTPUT_DIR,
    "embeddings"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    EMBEDDING_DIR,
    exist_ok=True
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ====================================================
# Initialize CSV
# ====================================================

COLUMNS = [

    "prompt_id",
    "run_id",

    "prompt",

    "prompt_char_length",
    "prompt_word_length",
    "prompt_token_length",

    "response",

    "response_char_length",
    "response_word_length",
    "response_token_length",

    "temperature",
    "top_p",
    "max_new_tokens",

    "runtime_latency",
    "tokens_per_second",

    "timestamp"
]

if not os.path.exists(CSV_PATH):

    pd.DataFrame(
        columns=COLUMNS
    ).to_csv(
        CSV_PATH,
        index=False
    )

# ====================================================
# Load completed records
# ====================================================

completed = set()

if os.path.exists(CSV_PATH):

    try:

        old_df = pd.read_csv(
            CSV_PATH,
            usecols=[
                "prompt_id",
                "run_id"
            ]
        )

        completed = set(
            zip(
                old_df["prompt_id"],
                old_df["run_id"]
            )
        )

        print(
            f"Loaded {len(completed)} completed samples."
        )

    except:

        pass

# ====================================================
# Load tokenizer
# ====================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====================================================
# Load model
# ====================================================

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

# ====================================================
# Load WildChat
# ====================================================

print("Loading WildChat...")

dataset = load_dataset(DATASET_LOCAL_PATH, split="train")
prompts = []

for item in dataset:

    try:

        conv = item["conversation"]

        if len(conv) == 0:
            continue

        first_turn = conv[0]

        if first_turn["role"] != "user":
            continue

        prompt = first_turn["content"].strip()

        if len(prompt) < 10:
            continue

        prompts.append(prompt)

    except:
        continue

print(
    f"Collected prompts: {len(prompts)}"
)

# ====================================================
# Sample prompts
# ====================================================

prompts = random.sample(
    prompts,
    min(NUM_PROMPTS, len(prompts))
)

# ====================================================
# Token length statistics
# ====================================================

prompt_infos = []

print(
    "Calculating prompt lengths..."
)

for idx, prompt in enumerate(
    tqdm(prompts)
):

    token_len = len(
        tokenizer.encode(
            prompt,
            add_special_tokens=False
        )
    )

    prompt_infos.append({

        "prompt_id": idx,

        "prompt": prompt,

        "prompt_char_length":
            len(prompt),

        "prompt_word_length":
            len(prompt.split()),

        "prompt_token_length":
            token_len
    })

# ====================================================
# Bucket Batching
# ====================================================

prompt_infos = sorted(
    prompt_infos,
    key=lambda x:
        x["prompt_token_length"]
)

# ====================================================
# Save Prompt Embeddings
# ====================================================

print(
    "Generating prompt embeddings..."
)

for start in tqdm(
    range(
        0,
        len(prompt_infos),
        BATCH_SIZE
    )
):

    batch = prompt_infos[
        start:
        start + BATCH_SIZE
    ]

    batch_prompts = [
        x["prompt"]
        for x in batch
    ]

    batch_ids = [
        x["prompt_id"]
        for x in batch
    ]

    need_compute = []

    need_indices = []

    for i, pid in enumerate(batch_ids):

        emb_path = os.path.join(
            EMBEDDING_DIR,
            f"prompt_{pid}.npy"
        )

        if not os.path.exists(
            emb_path
        ):
            need_compute.append(
                batch_prompts[i]
            )

            need_indices.append(i)

    if len(need_compute) == 0:
        continue

    inputs = tokenizer(
        need_compute,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        hidden = model.model(
            **inputs
        ).last_hidden_state

        embeddings = (
            hidden.mean(dim=1)
            .cpu()
            .numpy()
        )

    for local_idx, emb in enumerate(
        embeddings
    ):

        global_idx = (
            need_indices[
                local_idx
            ]
        )

        pid = batch_ids[
            global_idx
        ]

        np.save(

            os.path.join(
                EMBEDDING_DIR,
                f"prompt_{pid}.npy"
            ),

            emb
        )

# ====================================================
# Generate
# ====================================================

print(
    "\nStarting generation..."
)

for run_id in range(K):

    print(
        f"\n===== RUN {run_id+1}/{K} ====="
    )

    for start in tqdm(

        range(
            0,
            len(prompt_infos),
            BATCH_SIZE
        )

    ):

        batch = prompt_infos[
            start:
            start + BATCH_SIZE
        ]

        pending_batch = []

        for item in batch:

            key = (
                item["prompt_id"],
                run_id
            )

            if key in completed:
                continue

            pending_batch.append(
                item
            )

        if len(
            pending_batch
        ) == 0:
            continue

        batch_prompts = [

            x["prompt"]

            for x in pending_batch
        ]

        inputs = tokenizer(

            batch_prompts,

            return_tensors="pt",

            padding=True,

            truncation=True,

            max_length=
            MAX_INPUT_LENGTH
        )

        inputs = {

            k: v.to(
                model.device
            )

            for k, v
            in inputs.items()
        }

        input_lengths = (

            inputs[
                "attention_mask"
            ]

            .sum(dim=1)

            .tolist()
        )

        torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():

            outputs = model.generate(

                **inputs,

                do_sample=True,

                temperature=
                TEMPERATURE,

                top_p=
                TOP_P,

                max_new_tokens=
                MAX_NEW_TOKENS,

                pad_token_id=
                tokenizer.pad_token_id
            )

        torch.cuda.synchronize()

        end_time = time.time()

        batch_time = (
            end_time -
            start_time
        )

        decoded = tokenizer.batch_decode(

            outputs,

            skip_special_tokens=True
        )

        records = []

        for i, item in enumerate(
            pending_batch
        ):

            output_len = (

                outputs[i]
                .shape[0]

                - input_lengths[i]
            )

            latency = (
                batch_time
                /
                len(
                    pending_batch
                )
            )

            response = decoded[i]

            records.append({

                "prompt_id":
                    item[
                        "prompt_id"
                    ],

                "run_id":
                    run_id,

                "prompt":
                    item[
                        "prompt"
                    ],

                "prompt_char_length":
                    item[
                        "prompt_char_length"
                    ],

                "prompt_word_length":
                    item[
                        "prompt_word_length"
                    ],

                "prompt_token_length":
                    item[
                        "prompt_token_length"
                    ],

                "response":
                    response,

                "response_char_length":
                    len(
                        response
                    ),

                "response_word_length":
                    len(
                        response.split()
                    ),

                "response_token_length":
                    int(
                        output_len
                    ),

                "temperature":
                    TEMPERATURE,

                "top_p":
                    TOP_P,

                "max_new_tokens":
                    MAX_NEW_TOKENS,

                "runtime_latency":
                    latency,

                "tokens_per_second":
                    (
                        output_len
                        /
                        latency
                    )
                    if latency > 0
                    else 0,

                "timestamp":
                    pd.Timestamp.now()
            })

        # ====================
        # Immediate Save
        # ====================

        pd.DataFrame(
            records
        ).to_csv(

            CSV_PATH,

            mode="a",

            header=False,

            index=False
        )

        for r in records:

            completed.add(

                (
                    r["prompt_id"],
                    r["run_id"]
                )
            )

print(
    "\nDataset generation completed."
)

print(
    f"Saved to: {CSV_PATH}"
)
