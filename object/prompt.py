# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：prompt.py
@Author  ：Zhihao Zhang
@Date    ：2026/7/9 11:23 
@Describe：提示词对象
"""
import numpy as np


class Prompt:
    def __init__(self, prompt_id, prompt_text):
        """
        构建提示词对象
        :param prompt_id: 提示词ID
        :param prompt_text: 提示词文本内容
        """
        self.prompt_id = prompt_id
        self.prompt_text = prompt_text

        # 记录每次迭代输出的token长度
        self.output_token_lengths = []
        # 记录每次迭代输出的文本内容
        self.output_texts = []
        # 迭代次数
        self.repeat_times = 0
        # 是否稳定
        self.finished = -1

    def add_output_result(self, output_length, output_text=None):
        """
        添加一次推理结果
        :param output_length: 输出的token长度
        :param output_text: 输出的文本内容（可选）
        :return:
        """
        # 重复次数+1
        self.repeat_times += 1

        # 存储token长度
        self.output_token_lengths.append(output_length)
        self.output_texts.append(output_text)

    def is_stable(self, min_repeat_times, max_repeat_times, median_threshold,min_token_delta):
        """
        判断是否已经稳定
        :param min_repeat_times: 规定的最小迭代次数
        :param max_repeat_times: 规定的最大迭代次数
        :param median_threshold: 中位数阈值差异限制
        :param min_token_delta: 至少允许token差异数
        :return -1：未稳定； 0：达到最大上限结束； 1：因稳定早结束
        """
        if self.repeat_times >= max_repeat_times:
            return 0

        if self.repeat_times < min_repeat_times:
            return -1

        # 计算当前中位数和容忍阈值
        temp_median = np.median(self.output_token_lengths)
        delta = max(median_threshold * temp_median, min_token_delta)

        last1 = self.output_token_lengths[-1]
        last2 = self.output_token_lengths[-2]

        # 最近两次差异值小与阈值则稳定
        if abs(last1 - temp_median) <= delta and abs(last2 - temp_median) <= delta:
            return 1
        else:
            return -1

    def print_prompt(self):
        """
        打印当前prompt
        """
        print(f"{self.prompt_id}: {self.prompt_text}")

    def to_dict(self):
        """
        Prompt转dict，方便存储结果
        """
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "output_token_lengths": str(self.output_token_lengths),
            "output_texts": str(self.output_texts),
            "repeat_times": self.repeat_times,
            "finished": self.finished
        }
