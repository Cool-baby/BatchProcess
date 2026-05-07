# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：get_runtime_info.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 10:21 
@Describe：获取运行阶段GPU或KV缓存等设备信息
"""
import subprocess


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
