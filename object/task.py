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
        """
        self.task_id = task_id
        self.task_type = task_type
        self.prompt = prompt
        # 任务输入长度
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


    def cal_computation_density(self, m_param, l, h, h_kv, f_ser, bandwidth):
        """
        计算此任务的计算密度
        """
        # 计算 计算密度
        density = task_density(m_param, l, h, h_kv, self.input_length, self.predicted_output_length, f_ser, bandwidth)
        self.comp_density = density

        return density


    def predicted_output_length(self):
        """
        预测输出长度
        :return:
        """
        # TODO 增加模型输出长度预测功能
        self.predicted_output_length = random.randint(64, 256)
