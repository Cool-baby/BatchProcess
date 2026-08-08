# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/7 20:58
@Description: 工厂模式：调度器工厂
"""
from typing import Dict, Any
from core.scheduler.base_scheduler import BaseScheduler
from core.scheduler.bidirectional_scan_scheduler import BidirectionalScanScheduler
from core.scheduler.fcfs_scheduler import FCFSScheduler
from core.scheduler.prefix_tree_scheduler import PrefixTreeScheduler


class SchedulerFactory:
    """
    调度器工厂
    """
    _schedulers = {
        "fcfs": FCFSScheduler,
        "prefix_tree": PrefixTreeScheduler,
        "bilateral": BidirectionalScanScheduler,
    }

    @classmethod
    def create(cls, scheduler_name: str, config: Dict[str, Any]) -> BaseScheduler:
        scheduler_cls = cls._schedulers.get(scheduler_name.lower())
        if not scheduler_cls:
            raise ValueError(f"Unknown scheduler type: {scheduler_name}. Supported: {list(cls._schedulers.keys())}")
        return scheduler_cls(config)
