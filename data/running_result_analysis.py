# -*- coding: UTF-8 -*-
"""
@Project ：BatchProcess 
@File    ：running_result_analysis.py
@Author  ：Zhihao Zhang
@Date    ：2026/5/11 10:15 
@Describe：运行结果分析
"""


def analysis_tasks_result(all_tasks):
    """
    分析全部的任务信息
    :param all_tasks: 全部任务对象列表
    """
    print(f"\n------Task Summary------\n")
    # 总任务数量
    total_tasks = len(all_tasks)
    # 总完成任务数量
    total_finished_task = 0
    # 总截断任务数量
    total_truncated_task = 0
    # 总输出长度
    total_output_length = 0
    # 总推理时延
    total_time_cost = 0.0
    # 总预测和实际输出token数量差异
    total_diff_predict_actual_output = 0

    for task in all_tasks:
        if task.finished:
            total_finished_task += 1
        else:
            total_truncated_task += 1
        total_output_length += task.output_length
        total_time_cost += task.runtime_latency
        total_diff_predict_actual_output += abs(task.output_length - task.predicted_output_length)


    print(f"Total tasks: {total_tasks}, total runtime: {total_time_cost}, total output length: {total_output_length}, "
          f"avg. time cost: {total_time_cost / total_tasks:.4f}, avg. output length: {total_output_length / total_tasks:.4f}, "
          f"number of finished tasks: {total_finished_task} ({(total_finished_task / total_tasks)*100:.4f}%), "
          f"number of truncated tasks: {total_truncated_task}({(total_truncated_task / total_tasks)*100:.4f}%)")

    print(f"Avg. Difference between predicted and actual output token count: {total_diff_predict_actual_output / total_tasks:.4f} ")


def analysis_batches_result(all_batches):
    """
    分析全部的批次信息
    :param all_batches: 全部批次对象列表
    """
    print(f"\n------Batch Summary------\n")
    # 总批次数量
    total_batches = len(all_batches)
    # 总时间消耗
    total_time_cost = 0.0
    # 总token生成
    total_generated_tokens = 0

    for batch in all_batches:
        total_time_cost += batch.time_cost
        total_generated_tokens += batch.total_generated_tokens

    print(f"Total batches: {total_batches}, total time cost: {total_time_cost} s, total generated tokens: {total_generated_tokens}, "
          f"avg. time cost: {total_time_cost / total_batches:.4f} s, avg. generated tokens: {total_generated_tokens / total_batches:.4f}, "
          f"throughput: {total_generated_tokens / total_time_cost:.4f} tokens/sec")
