# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：inference_engine.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/8 16:37 
@Describe： 推理引擎
"""
import logging
import time
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class InferenceEngine:
    def __init__(self, model_name, model_path, max_new_tokens=1024, max_input_length=2048):
        """
        初始化推理引擎
        :param model_name: 模型名称
        :param model_path: model本地位置
        :param max_new_tokens: 最多生成多少个新 token
        :param max_input_length: 最大的输入token数量
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            logging.info("No GPU found, using CPU")
            # raise Exception("CUDA not available")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_length = max_input_length

        # tokenizer
        self.tokenizer = None
        logging.info("Loading tokenizer...")
        if model_name == "Qwen2.5-3B-Instruct" or model_name == "Llama-3.2-3B-Instruct":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        elif model_name == "Gemma-3-4b-Instruct":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # 避免 pad_token 与 eos_token 相同导致的早停失效
            if "<pad>" in self.tokenizer.get_vocab():
                self.tokenizer.pad_token = "<pad>"
            elif self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        else:
            raise Exception("Unknown model tokenizer")

        # model
        logging.info(f"Loading model - {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

        self.model.eval()
        logging.info(self.model.hf_device_map)


    def get_prompt_token(self, prompt):
        """
        计算输入prompt的token长度
        :param prompt: 提示词
        :return: token
        """
        return self.tokenizer.encode(prompt, add_special_tokens=False)


    def build_prompts(self, batch):
        """
        统一 batch prompt 生成（避免逐个 tokenize）
        """

        texts = []

        for task in batch:
            messages = [
                {"role": "user", "content": task.prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            texts.append(text)

        return texts


    def inference_batch(self, batch):

        # 1. 取出prompt文本
        batch_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": task.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for task in batch.tasks
        ]

        # 2. tokenizer
        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length  # 限制最大的Prompt输入长度，避免OOM
        )

        # 将所有Tensor移动到GPU
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # 计算每个Prompt自己的输入长度
        input_seq_len = inputs["input_ids"].shape[1]

        # 3. generate
        if self.model_name == "Qwen2.5-3B-Instruct" or self.model_name == "Llama-3.2-3B-Instruct":
            start_time = time.time()
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            end_time = time.time()
            batch_latency = end_time - start_time
            batch.time_cost = batch_latency

            # 4. 更新Prompt对象
            for idx, task in enumerate(batch.tasks):
                # 获取模型新生成的 Token
                generated_ids_with_pad = outputs[idx][input_seq_len:]
                eos_positions = (generated_ids_with_pad == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    first_eos_idx = eos_positions[0].item()
                    # 截取到第一个 eos_token 为止（不包含eos本身，如果你想算入结束符，则改为 first_eos_idx + 1）
                    true_generated_ids = generated_ids_with_pad[:first_eos_idx]
                else:
                    # 如果没生成 eos（即跑满了 max_new_tokens），为安全起见去除可能的 pad
                    true_generated_ids = generated_ids_with_pad[generated_ids_with_pad != self.tokenizer.pad_token_id]
                # 解码生成文本
                output_text = self.tokenizer.decode(true_generated_ids, skip_special_tokens=True)
                # 输出 Token 数量
                output_length = len(true_generated_ids)
                # 更新 Task 对象
                task.output = output_text
                task.runtime_latency = batch_latency
                task.output_length = output_length
                task.finished = 1
                batch.total_generated_tokens += output_length
        elif self.model_name == "Gemma-3-4b-Instruct":
            # 获取 Gemma-3 对话模式下的所有结束符 (包含 <eos> 和 <end_of_turn>)
            eos_token_ids = [self.tokenizer.eos_token_id]
            end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_of_turn_id is not None and end_of_turn_id != self.tokenizer.unk_token_id:
                eos_token_ids.append(end_of_turn_id)
            # 收集所有的停止 Token ID 用于后续精确切片判定
            stop_token_ids = set(eos_token_ids + [self.tokenizer.pad_token_id])

            start_time = time.time()
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=eos_token_ids,  # 传入包含 <end_of_turn> 的列表
                )
            end_time = time.time()
            batch_latency = end_time - start_time
            batch.time_cost = batch_latency

            # 4. 更新 Prompt 对象
            for idx, task in enumerate(batch.tasks):
                # 切除 Prompt 部分，仅保留新生成的 Token
                generated_sequence = outputs[idx][input_seq_len:].tolist()

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
                output_text = self.tokenizer.decode(true_generated_ids, skip_special_tokens=True)
                # 真实的输出 Token 数量
                output_length = len(true_generated_ids)

                # 更新 Task 对象
                task.output = output_text
                task.runtime_latency = batch_latency
                task.output_length = output_length
                task.finished = 1
                batch.total_generated_tokens += output_length

        # 计算当前batch的token吞吐量
        batch.tokens_throughput = batch.total_generated_tokens / batch.time_cost
