# -*- coding: utf-8 -*-

"""
@Author: Zhihao Zhang
@Email: zhang_zhi_hao@foxmail.com
@Date: 2026/8/7 20:58
@Description: 工厂模式：调度器工厂
"""
from typing import Dict, Any
from core.scheduler.ablation_scheduler import BidirectionalScanSchedulerAblation
from core.scheduler.base_scheduler import BaseScheduler
from core.scheduler.bidirectional_scan_scheduler import BidirectionalScanScheduler
from core.scheduler.bilateral_scheduler_wo_med import BidirectionalScanSchedulerWoMed
from core.scheduler.bilateral_scheduler_wo_p95 import BidirectionalScanSchedulerWoP95
from core.scheduler.blendserve_scheduler import BlendServeScheduler
from core.scheduler.dftsp_scheduler import DFTSPScheduler
from core.scheduler.fcfs_scheduler import FCFSScheduler
from core.scheduler.optimized_bilateral_scheduler import BidirectionalScanSchedulerPro
from core.scheduler.prefix_tree_scheduler import PrefixTreeScheduler
from core.scheduler.ranking_scheduler import RankAntiStarvationScheduler
from core.scheduler.sjf_scheduler import SJFScheduler


class SchedulerFactory:
    """
    调度器工厂
    """
    _schedulers = {
        "fcfs": FCFSScheduler,
        "prefix-tree": PrefixTreeScheduler,
        "bilateral": BidirectionalScanScheduler,
        "stabs": BidirectionalScanSchedulerPro,
        "stabs-wo-med": BidirectionalScanSchedulerWoMed,
        "stabs-wo-p95": BidirectionalScanSchedulerWoP95,
        "sjf": SJFScheduler,
        "dftsp": DFTSPScheduler,
        "ranking": RankAntiStarvationScheduler,
        "blendserve": BlendServeScheduler,
        "stabs-wo-prf": BidirectionalScanSchedulerAblation,
    }

    @classmethod
    def create(cls, scheduler_name: str, config: Dict[str, Any]) -> BaseScheduler:
        scheduler_cls = cls._schedulers.get(scheduler_name.lower())
        if not scheduler_cls:
            raise ValueError(f"Unknown scheduler type: {scheduler_name}. Supported: {list(cls._schedulers.keys())}")
        return scheduler_cls(config)
