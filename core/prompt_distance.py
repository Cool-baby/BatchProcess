# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：prompt_distance.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/7 11:31 
@Describe：计算两个任务在图中距离，反映前缀共享潜力
"""


def common_prefix_length(p1: str, p2: str):
    """
    计算Prompt p1和Prompt p2之间的共享前缀长度
    :param p1: Prompt 1
    :param p2: Prompt 2
    :return: 共享前缀长度
    """
    length = 0

    for c1, c2 in zip(p1, p2):

        if c1 == c2:
            length += 1
        else:
            break

    return length


def prefix_distance(p1: str, p2: str):
    """
    计算Prompt p1和Prompt p2在前缀图中的距离
    :param p1: Prompt 1
    :param p2: Prompt 2
    :return:
    """
    # 计算共享前缀长度
    cp = common_prefix_length(p1, p2)

    # Prompt 1 和 Prompt 2长度
    p1_length = len(p1)
    p2_length = len(p2)

    return 1.0 - cp / max(p1_length, p2_length)
