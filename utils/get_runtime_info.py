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


def start_gpu_monitor(device_id=0, interval=0.1):
    """启动子线程进行GPU高频采样"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    ctx = {'run': True, 'util': [], 'mem': []}

    def sample_loop():
        while ctx['run']:
            ctx['util'].append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            ctx['mem'].append(pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 3))  # 转换为GB
            time.sleep(interval)

    t = threading.Thread(target=sample_loop)
    t.start()
    return ctx, t


def stop_gpu_monitor(monitor_ctx):
    """停止采样并计算平均值"""
    ctx, t = monitor_ctx
    ctx['run'] = False
    t.join()
    avg_util = sum(ctx['util']) / len(ctx['util']) if ctx['util'] else 0
    avg_mem = sum(ctx['mem']) / len(ctx['mem']) if ctx['mem'] else 0
    logging.info(f"当前批次平均计算利用率: {avg_util:.1f}%, 平均显存: {avg_mem:.2f} GB")
    return avg_util, avg_mem
