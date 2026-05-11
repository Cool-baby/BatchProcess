# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：task.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 11:24 
@Describe：Task对象
"""
import random
from core.computation_density import task_density


class Task:
    def __init__(self, task_id, task_type, prompt):
        """
        推理任务
        :param task_id: 任务ID
        :param task_type: 任务类型
        :param prompt: 任务输入的提示词
        :param prompt_token: tokenizer之后的prompt
        """
        self.task_id = task_id
        self.task_type = task_type
        self.prompt = prompt
        # 任务输入token长度
        self.input_length = 0
        # 任务计算密度
        self.comp_density = 0
        # 任务输出结果
        self.output = None
        # 任务实际输出长度
        self.output_length = 0
        # 任务预估输出长度
        self.predicted_output_length = 0
        # 任务所需GPU计算资源
        self.comp_cost = 0
        # 任务所需GPU内存资源
        self.mem_cost = 0

        # 推理结果
        self.runtime_latency = None
        # 是否推理完成 0 未完成，1 完成
        self.finished = 0

    def cal_computation_density(self, m_param, l, h, h_kv, f_ser, bandwidth):
        """
        计算此任务的计算密度
        """
        # 计算 计算密度
        density = task_density(m_param, l, h, h_kv, self.input_length, self.predicted_output_length, f_ser, bandwidth)
        self.comp_density = density

        return density


    def predicted_output(self):
        """
        预测输出长度
        :return:
        """
        # TODO 增加模型输出长度预测功能
        self.predicted_output_length = random.randint(64, 256)


    def print_task(self):
        """
        打印当前任务
        """
        print(f"Task ID: {self.task_id}, Task Type: {self.task_type}, Prompt: {self.prompt}, Response:{self.response}, "
              f"input length: {self.input_length}, output length:{self.output_length}, predicted output length: {self.predicted_output_length}, "
              f"computation density: {self.comp_density}")

    def print_task_concise(self):
        """
        打印任务简略信息
        """
        print(f"Task ID: {self.task_id}, Task Type: {self.task_type}, "
              f"input length: {self.input_length}, output length:{self.output_length}, predicted output length: {self.predicted_output_length}, "
              f"computation density: {self.comp_density}")


    def to_dict(self):
        """
        Task转dict，方便存储结果
        """
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "prompt": self.prompt,

            "input_length": self.input_length,
            "comp_density": self.comp_density,

            "output": self.output,
            "output_length": self.output_length,
            "predicted_output_length": self.predicted_output_length,

            "comp_cost": self.comp_cost,
            "mem_cost": self.mem_cost,

            "runtime_latency": self.runtime_latency,
            "finished": self.finished
        }


def generate_task_by_prompt(dataset_name, dataset_type, idx, single_data):
    """
    基于数据集中的单个数据生成一个任务对象
    :param dataset_name: 数据集名称
    :param dataset_type: 任务类型
    :param idx: 任务序号
    :param single_data: 数据集中的单个数据
    :return: Task Object
    """
    # 如果是WildChat对话
    if dataset_name == "allenai/WildChat":
        # 提取此对话prompt
        conversation = single_data["conversation"]
        if len(conversation) == 0:
            return None
        prompt = conversation[0]["content"]

        # 空prompt过滤
        if prompt is None or len(prompt.strip()) == 0:
            return None
    else:
        raise Exception(f"Dataset {dataset_name} not implemented")

    return Task(idx, dataset_type, prompt)
