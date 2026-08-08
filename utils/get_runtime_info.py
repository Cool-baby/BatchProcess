# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：get_runtime_info.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 10:21 
@Describe：获取运行阶段GPU或KV缓存等设备信息
"""
import logging
import subprocess
import threading, time
from contextlib import contextmanager
try:
    import pynvml
except ImportError:
    logging.info("Import pynvml failed")
    pass


def get_gpu_metrics():
    """
    使用NVIDIA-smi获取GPU资源利用率和显存信息
    """
    # 使用NVIDIA-smi获取GPU资源利用率和显存信息
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=utilization.gpu,memory.used,memory.total',
            '--format=csv,nounits,noheader'
        ],
        stdout=subprocess.PIPE,
        text=True
    )
    gpu_util, mem_used, mem_total = result.stdout.strip().split(", ")

    return {
        "gpu_util": float(gpu_util),
        "mem_used": float(mem_used),
        "mem_total": float(mem_total)
    }


def get_kv_cache_size(past_key_values):
    """
    获取KV缓存占用大小
    :param past_key_values: past_key_values 是 Hugging Face Transformers 库中缓存历史 token 的 Key 和 Value 向量‌
    :return: KV缓存占有（MB）
    """
    if past_key_values is None:
        return 0

    total = 0
    for layer in past_key_values:
        k, v = layer
        total += k.numel() * k.element_size()
        total += v.numel() * v.element_size()

    return total / (1024 ** 2)  # MB


@contextmanager
def track_gpu_resources(device_id=0, interval=0.1):
    """
    基于上下文管理器的 GPU 资源监控器 (支持提取统计数据)
    """
    # 初始化统计字典
    ctx = {'run': True, 'util': [], 'mem': [], 'avg_util': 0.0, 'avg_mem': 0.0}

    if pynvml is None:
        yield ctx
        return

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    def sample_loop():
        while ctx['run']:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 3) # GB
                ctx['util'].append(util)
                ctx['mem'].append(mem)
            except Exception:
                break
            time.sleep(interval)

    # 启动守护线程
    t = threading.Thread(target=sample_loop, daemon=True)
    t.start()

    try:
        # 将 ctx 字典引用传递给 with ... as 变量
        yield ctx
    finally:
        # 1. 停止线程
        ctx['run'] = False
        t.join(timeout=1.0)
        pynvml.nvmlShutdown()

        # 2. 计算平均值并回写到 ctx 字典中
        if ctx['util']:
            ctx['avg_util'] = sum(ctx['util']) / len(ctx['util'])
            ctx['avg_mem'] = sum(ctx['mem']) / len(ctx['mem'])
