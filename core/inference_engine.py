# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：inference_engine.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/8 16:37 
@Describe： 推理引擎
"""
import time
import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class InferenceEngine:
    def __init__(self, model_path, max_new_tokens=128):
        """
        初始化推理引擎
        :param model_path: model本地位置
        :param max_new_tokens: 最多生成多少个新 token
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 避免 pad warning
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # model
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map="auto", trust_remote_code=True)

        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

        self.model.eval()


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


    def tokenize_prompt(self, prompt):
        """
        tokenizer 处理 prompt
        :param prompt: 输入的prompt
        :return: formatted_prompt, input_ids, attention_mask, input_length
        """
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        encoded = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, add_special_tokens=False)

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        input_length = input_ids.shape[0]

        return formatted_prompt, input_ids, attention_mask, input_length


    def inference_batch(self, batch):

        tasks_batch = batch.tasks
        texts = self.build_prompts(tasks_batch)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        prompt_length = inputs["input_ids"].shape[1]

        # GPU timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        batch_latency = end_time - start_time
        batch.time_cost = batch_latency

        for i, task in enumerate(tasks_batch):
            generated_tokens = outputs[i][prompt_length:]
            # 找第一个EOS结束位置
            eos_positions = (generated_tokens == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]

            if len(eos_positions) > 0:
                real_length = eos_positions[0].item()
                generated_tokens = generated_tokens[:real_length]
                finished = True
                batch.finished_tasks += 1
            else:
                real_length = generated_tokens.shape[0]
                finished = False
                batch.truncated_tasks += 1

            # decode
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)

            # 去除user/assistant标记
            for tag in ["<｜User｜>", "<｜Assistant｜>", "<|User|>", "<|Assistant|>", "<｜end▁of▁sentence｜>", "</s>", "<s>", "<eos>"]:
                response = response.replace(tag, "")

            response = response.strip()

            # 提取think之后内容
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            # 保存结果
            task.output = response
            task.runtime_latency = batch_latency
            task.output_length = real_length
            task.finished = 1 if finished else 0
            batch.total_generated_tokens += real_length

        # 计算当前batch的token吞吐量
        batch.tokens_throughput = batch.total_generated_tokens / batch.time_cost

        return batch_latency
