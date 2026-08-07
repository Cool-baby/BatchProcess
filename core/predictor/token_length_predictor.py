# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/7/29 20:29
@Description: token输出长度预测器
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


# 模型定义
class _LLMLengthPredictorNet(nn.Module):
    def __init__(self, model_name_or_path: str, peft_config=None):
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

        # Median 升级为独立的 MLP
        self.median_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1, dtype=torch.bfloat16)
        )

        # P95 修改为同构的独立 MLP
        self.p95_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1, dtype=torch.bfloat16)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
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

        # 【修改点 3】：两个头完全独立预测
        median_pred = self.median_head(last_token_hidden_states)
        p95_pred = self.p95_head(last_token_hidden_states)

        return torch.cat([median_pred, p95_pred], dim=-1)


# 2. 面向外部调用的预测器类
class TokenLengthPredictor:
    def __init__(
            self,
            base_model_path: str,
            checkpoint_path: str,
            max_length: int = 2048,
            device: str = None
    ):
        """
        初始化预测器
        :param base_model_path: 本地基础模型路径 (如 ./models/Qwen2.5-1.5B)
        :param checkpoint_path: 训练好的权重路径 (如 ./length_predictor_outputs/final_model_0729.pt)
        :param max_length: 文本最大截断长度
        :param device: 推理设备 ('cuda' 或 'cpu')
        """
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 构建模型并加载训练权重
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=True,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = _LLMLengthPredictorNet(base_model_path, peft_config=peft_config)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def _preprocess_single(self, prompt_text: str, inference_model: str) -> List[int]:
        """单文本拼接与头尾保留截断（安全拼接版本）"""
        prefix = f"[INFERENCE MODEL]: {inference_model}\n[PROMPT]:\n"
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

        bos_len = 1 if self.tokenizer.bos_token_id is not None else 0
        eos_len = 1 if self.tokenizer.eos_token_id is not None else 0
        remain_len = self.max_length - len(prefix_tokens) - bos_len - eos_len

        text_tokens = self.tokenizer.encode(str(prompt_text), add_special_tokens=False)

        if len(text_tokens) > remain_len:
            half = remain_len // 2
            text_tokens = text_tokens[:half] + text_tokens[-half:]

        final_tokens = []
        if self.tokenizer.bos_token_id is not None:
            final_tokens.append(self.tokenizer.bos_token_id)

        final_tokens.extend(prefix_tokens)
        final_tokens.extend(text_tokens)

        if self.tokenizer.eos_token_id is not None:
            final_tokens.append(self.tokenizer.eos_token_id)

        return final_tokens

    def predict_one(self, prompt_text: str, inference_model: str) -> Dict[str, float]:
        """预测单个 Prompt 的 Token 长度"""
        res = self.predict_batch([{"prompt_text": prompt_text, "inference_model": inference_model}])
        return res[0]

    def predict_batch(
            self,
            items: List[Dict[str, str]],
            batch_size: int = 16
    ) -> List[Dict[str, float]]:
        """
        批量预测 Token 长度
        :param items: [{'prompt_text': '...', 'inference_model': '...'}, ...]
        :param batch_size: 批处理大小
        :return: [{'median_length': 150.2, 'p95_length': 185.6}, ...]
        """
        results = []

        with torch.no_grad():
            for i in range(0, len(items), batch_size):
                batch_items = items[i: i + batch_size]

                # Tokenize 批次
                input_ids_list = [
                    torch.tensor(self._preprocess_single(item["prompt_text"], item["inference_model"]))
                    for item in batch_items
                ]

                # 手动 Padding 齐平
                input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                    input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).to(self.device)

                attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long().to(self.device)

                # 推理
                outputs = self.model(input_ids_padded, attention_mask)

                # 反 Log 变换还原: expm1(x) = exp(x) - 1
                medians = torch.expm1(outputs[:, 0].float()).cpu().numpy()
                p95s = torch.expm1(outputs[:, 1].float()).cpu().numpy()

                for m, p in zip(medians, p95s):
                    # 【修改点 4】：强化后处理约束。由于长尾分布特性，P95理应比Median大一定比例。
                    # 直接强制 P95 >= Median * 1.1，而不仅仅是防止 P95 < Median
                    adjusted_p = max(m * 1.1, p)

                    # 仅在调试时可以保留此打印，如果批量大建议注释掉以免刷屏
                    if adjusted_p != p:
                        pass  # print(f"WARNING: P95 Constrained. Original: {p:.1f}, Median: {m:.1f}, Adjusted: {adjusted_p:.1f}")

                    results.append({
                        "median_length": float(np.round(m, 1)),
                        "p95_length": float(np.round(adjusted_p, 1))
                    })

        return results


if __name__ == "__main__":
    # 路径配置
    LOCAL_MODEL_PATH = "/Users/hao/Workspace/Other/models/Qwen2.5-1.5B"
    CHECKPOINT_PATH = "./final_model_0731.pt"

    predictor = TokenLengthPredictor(base_model_path=LOCAL_MODEL_PATH, checkpoint_path=CHECKPOINT_PATH, max_length=2048)

    prompt1 = "There were 21 dollars in the cookie jar. Doris spent $6 from the cookie jar. Martha spent half as much as Doris. How much money, in dollars, was left in the cookie jar?"
    # inference_mode1 = "Gemma-3-4b-Instruct"
    # inference_mode1 = "Llama-3.2-3B-Instruct"
    inference_mode1 = "Qwen2.5-3B-Instruct"
    res = predictor.predict_one(prompt1, inference_mode1)
    print(f"Result: {res}, median: {res['median_length']}, p95: {res['p95_length']}")
