# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：computation_density.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 10:34 
@Describe：计算密度，包含计算密集型和内存密集型计算方法
"""
LAMBDA = 4  # 内存密集型任务超参数

def comp_density(m_param, input_length, predicted_output_length, h, l, f_ser):
    """
    计算密集型任务估算
    """
    return (2 * m_param * (input_length + predicted_output_length) + 4 * input_length * input_length * h * l) / f_ser


def mem_density(l, h_kv, input_length, predicted_output_length, bandwidth):
    """
    内存密集型任务估算
    """
    return (LAMBDA * l * h_kv * (input_length * predicted_output_length + 0.5 * predicted_output_length * predicted_output_length)) / bandwidth


def task_density(m_param, l, h, h_kv, input_length, predicted_output_length, f_ser, bandwidth):
    """
    计算任务密度，较高计算密度对应着计算密集型推理任务,而较低计算密度则表示内存密集型推理任务
    :param m_param: 大规模模型的参数量
    :param l: 大规模模型层数
    :param h: 每层隐藏维度
    :param h_kv: KV头特征维度
    :param input_length: 任务输入长度
    :param predicted_output_length: 预测输出长度
    :param f_ser: 服务节点的浮点计算能力
    :param bandwidth: GPU 的内存带宽
    :return: 计算密度
    """
    comp_dens = comp_density(m_param, input_length, predicted_output_length, h, l, f_ser)
    mem_dens = mem_density(l, h_kv, input_length, predicted_output_length, bandwidth)

    return comp_dens / mem_dens
