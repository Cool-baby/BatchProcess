# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/7/28 20:24
@Description: 训练token长度预测器
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import functools


# 1. 模型架构定义 (Backbone + 自定义回归头)
class LLMLengthPredictor(nn.Module):
    def __init__(self, model_name_or_path, peft_config=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True
        )

        if peft_config:
            self.backbone = get_peft_model(self.backbone, peft_config)

        hidden_size = self.backbone.config.hidden_size

        # 增加网络深度，让 Median 头具备更强拟合能力
        self.median_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1, dtype=torch.bfloat16)
        )

        # P95 需要拟合更复杂的长尾边界，同样给予完整的 MLP 结构
        self.p95_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1, dtype=torch.bfloat16)
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden_states = outputs.last_hidden_state[
            torch.arange(batch_size, device=input_ids.device),
            sequence_lengths
        ]

        # 两个头独立预测，自然输出。如果推理时 P95 < Median，可以做后处理 clamp，但在训练时让它们自由拟合。
        median_pred = self.median_head(last_token_hidden_states)
        p95_pred = self.p95_head(last_token_hidden_states)

        return torch.cat([median_pred, p95_pred], dim=-1)


# 2. 自定义 Trainer 与 Pinball Loss
def quantile_loss(preds, targets, quantiles=[0.5, 0.95], p95_weight_scale=2.0):
    """
    标准的严格分位数损失 (Strict Pinball Loss)
    preds: [batch_size, 2]
    targets: [batch_size, 2]
    """
    loss = 0.0
    # 为不同分位数分配全局重要性权重，强迫模型关注 P95
    global_weights = [1.0, p95_weight_scale]

    for i, q in enumerate(quantiles):
        error = targets[:, i] - preds[:, i]

        # 标准 Pinball Loss 公式 max(q * error, (q-1) * error)
        q_loss = torch.where(error > 0, q * error, (q - 1) * error)

        loss += (global_weights[i] * q_loss).mean()

    return loss


class QuantileRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # 由于这里用的是自定义模块而不是 HF 标准模型结构，因此输出就是 preds
        preds = model(**inputs)

        # 如果 finished == 0 (数据被截断)，可以通过 label 中的第三维传入掩码来动态调整权重
        # 此处简化为直接算 Loss
        loss = quantile_loss(preds, labels)

        return (loss, preds) if return_outputs else loss


# 3. 数据集与预处理 (头尾保留截断策略)
class PromptLengthDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=2048):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        prefix = f"[INFERENCE MODEL]: {row['inference_model']}\n[PROMPT]:\n"
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

        # 预留 BOS 和 EOS 的位置
        bos_len = 1 if self.tokenizer.bos_token_id is not None else 0
        eos_len = 1 if self.tokenizer.eos_token_id is not None else 0
        remain_len = self.max_length - len(prefix_tokens) - bos_len - eos_len

        prompt_text = str(row["prompt_text"])
        text_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        if len(text_tokens) > remain_len:
            half = remain_len // 2
            text_tokens = text_tokens[:half] + text_tokens[-half:]

        # 【核心修复】：安全拼接 Token，避免加入 None
        final_tokens = []
        if self.tokenizer.bos_token_id is not None:
            final_tokens.append(self.tokenizer.bos_token_id)

        final_tokens.extend(prefix_tokens)
        final_tokens.extend(text_tokens)

        if self.tokenizer.eos_token_id is not None:
            final_tokens.append(self.tokenizer.eos_token_id)

        # 目标值转换
        median_target = np.log1p(row["median_length"])
        p95_target = np.log1p(row["p95_length"])
        labels = torch.tensor([median_target, p95_target], dtype=torch.float)

        return {
            "input_ids": final_tokens,
            "labels": labels
        }


def collate_fn(batch, pad_token_id=0):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    labels = torch.stack([item["labels"] for item in batch])

    # 填充对齐
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    attention_mask = (input_ids_padded != pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "labels": labels
    }


def load_dataset_from_jsonl(file_path, tokenizer, max_length):
    """辅助函数：读取 jsonl 并转换为 PromptLengthDataset"""
    records = []
    buffer = ""

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            buffer += line
            try:
                # 尝试解析当前累积的文本
                obj = json.loads(buffer)
                records.append(obj)
                buffer = ""  # 解析成功，清空缓冲区，准备读取下一个对象
            except json.JSONDecodeError:
                # 解析失败，说明当前 JSON 对象由于换行还没读取完整，继续读取下一行拼接到 buffer
                continue

    if buffer.strip():
        print(f"警告: 文件末尾存在未完整解析的数据: {buffer[:50]}...")

    df = pd.DataFrame(records)

    # 提取嵌套字段
    df['median_length'] = df['statistics'].apply(lambda x: x.get('median_length', 0))
    df['p95_length'] = df['statistics'].apply(lambda x: x.get('p95_length', 0))

    print(f"成功加载文件 {file_path}，有效数据共 {len(df)} 条。")
    return PromptLengthDataset(df, tokenizer, max_length)


# 4. 主执行流程
def main():
    # 参数设置
    MODEL_PATH = "/HOME/scw6e0x/run/zzhao/models/Qwen2.5-1.5B"  # 本地模型文件夹路径
    MAX_LENGTH = 2148
    EPOCHS = 3

    # 数据集路径
    TRAIN_JSONL_PATH = "./data/MoLD_shuffled_train.jsonl"  # 全量训练集路径
    TEST_JSONL_PATH = "./data/MoLD_shuffled_test.jsonl"  # 独立测试集路径 (如果不打算在训练中评估，可设为 None)

    print("1. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("2. 准备数据集...")
    # 全量数据作为训练集
    train_dataset = load_dataset_from_jsonl(TRAIN_JSONL_PATH, tokenizer, MAX_LENGTH)

    # 加载独立测试集（如果存在）
    eval_dataset = None
    if TEST_JSONL_PATH and os.path.exists(TEST_JSONL_PATH):
        eval_dataset = load_dataset_from_jsonl(TEST_JSONL_PATH, tokenizer, MAX_LENGTH)

    print("3. 初始化模型 (LoRA)...")
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = LLMLengthPredictor(MODEL_PATH, peft_config=peft_config)
    model.backbone.print_trainable_parameters()

    print("4. 配置训练参数...")
    # 根据是否有测试集决定是否开启 eval_strategy
    has_eval = eval_dataset is not None

    training_args = TrainingArguments(
        output_dir="./length_predictor_outputs",
        learning_rate=3e-4,

        # 核心优化 1：降低物理 Batch Size，开启梯度累加
        # 等效的总 Batch Size 依然是 2 * 4 = 8，但单次前向传播只吃 1/4 的显存！
        per_device_train_batch_size=2,  # 从原先的 8 降低到 2
        gradient_accumulation_steps=4,  # 每 4 步更新一次梯度
        per_device_eval_batch_size=4,  # 评估时的 Batch Size 也适当调小

        # 核心优化 2：开启梯度检查点 (以微小计算时间换取巨额显存节省)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 推荐针对 PyTorch 2.x 设置

        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        load_best_model_at_end=has_eval,
        logging_steps=10,

        # 保证开启混合精度训练
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        remove_unused_columns=False,
        dataloader_num_workers=2,  # 多进程加载数据提升速度
    )

    # 动态绑定正确的 pad_token_id
    custom_collate_fn = functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    trainer = QuantileRegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_collate_fn,
    )

    print("5. 开始全量训练...")
    trainer.train()

    # 6. 保存最终模型
    output_dir = "./length_predictor_outputs"
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 保证文件夹已存在时不报错

    save_path = os.path.join(output_dir, "STABS_prod_model_0806.pt")

    print(f"6. 正在保存权重到: {save_path} ...")
    torch.save(model.state_dict(), save_path)
    print(f"训练完成！全量训练模型已成功保存至: {save_path}")


if __name__ == "__main__":
    main()
