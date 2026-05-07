# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：task.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 11:24 
@Describe：Task对象
"""


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
        self.task_output = 0


    def get_computation_density(self):
        """
        获取任务计算密度
        :return:
        """
        pass