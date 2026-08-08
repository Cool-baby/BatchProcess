# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/8 10:10
@Description: 抽象基类 BaseScheduler，全部调度器都继承此类，并实现 schedule() 方法
"""
from abc import ABC, abstractmethod
from object.batch import Batch
from object.task import Task
from typing import List, Dict, Any


class BaseScheduler(ABC):
    def __init__(self, config: Dict[str, Any]):
        """
        抽象基类 BaseScheduler
        """
        self.config = config

    @abstractmethod
    def schedule(self, tasks: List[Task], **kwargs) -> List[Batch]:
        """
        接收任务列表，返回切分好的 Batch 列表，各调度器在此方法内自行完成所需的操作
        """
        pass
