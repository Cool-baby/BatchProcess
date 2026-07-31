# Batch Task Scheduling and Batch Processing Optimization Method Based on Regional Cloud

--- 

## Environment
To set up the environment, run the following commands:
```
conda create -n for_batch_proc python==3.12
conda activate for_batch_proc
bash requirement.sh
```

## Hyperparameter Settings
```
# LLM超参数
MODEL_PATH = "path"
M_PARAM=7e9  # 大规模模型的参数量
LLM_L=32  # 大规模模型层数
LLM_H=4096  # 每层隐藏维度
LLM_H_KV=128  # KV头特征维度
MAX_NEW_TOKENS = 1024  # 最多生成多少个新 token

# 服务器超参数
SER_GPU_FP = 22e12  # 服务节点的浮点计算能力(FLOPS)
SER_GPU_MEMORY = 16 * 1024  # GPU 的最大内存(MB)
SER_GPU_BANDWIDTH = 288e9  # GPU 的内存带宽
SER_NUMBER = 2  # 可用服务器数量，这决定了谱聚类的聚类数量

# 数据集超参数
DATASET_NAME = "allenai/WildChat"  # 数据集名称
DATASET_LOCAL_PATH = "path"  # 数据集本地路径
DATASET_TYPE = "chat"  # 数据集类型
DATASET_SELECT_SEED = 43  # 数据集抽取随机种子

# 批量大小超参数
SAMPLE_SIZE = 500  # 每次总共产生多少任务
MAX_SINGLE_BATCH_SIZE = 16  # 最大单批次大小
SCHEDULER = "bilateral"  # 批处理任务调度器
```

## Run
```
python main.py
```

---
## MoLD: Multi-model Output Length Distribution Dataset

### Overview

MoLD (**Multi-model Output Length Distribution Dataset**) is a benchmark dataset designed for studying **robust LLM output token length prediction**.

Existing output length prediction methods usually formulate the task as a deterministic regression problem, where each prompt is associated with only one observed output length. However, the generation process of large language models (LLMs) is inherently stochastic, and the same prompt may produce outputs with significantly different lengths across independent generations. Moreover, different LLMs may exhibit distinct output length behaviors for the same input prompt.

To address these challenges, MoLD constructs a **prompt-conditioned output length distribution dataset** by repeatedly querying the same prompt on the same LLM and recording the generated token lengths. Instead of assigning a single output length label, each sample contains an empirical output length distribution together with robust statistical descriptors, enabling research on distribution-aware and uncertainty-aware output length prediction.

---

### Dataset Statistics

MoLD contains repeated inference results from multiple representative open-source LLMs across diverse tasks.

| Category | Description                                   |
|----------|-----------------------------------------------|
| Number of prompts | Approximately 14K                             |
| Number of LLMs | 3 (Qwen2.5-3B, Llama-3.2-3B, Gemma-3-4B)      |
| Task categories | Chat, Math, Long-sequence                     |
| Source datasets | WildChat, GSM8K, LongBench                    |
| Inference models | Gemma-3-4B, Llama-3.2-3B-Instruct, Qwen2.5-3B |
| Annotation type | Prompt-conditioned output length distribution |
| Prediction targets | Median output length and P95 output length    |

---

### Data Format

MoLD is provided in JSONL format.

Each line represents one prompt-model sample.

Example:

```json
{
  "sample_id": "xxxxxxxxxxxxxxxx",
  "prompt_id": "b34c99c2e971bca3",
  "prompt_text": "How to write the OOO mail very professionally",
  "dataset_name": "WildChat",
  "task_type": "chat",
  "inference_model": "Llama-3.2-3B-Instruct",
  "output_token_lengths": [630, 512, 511],
  "repeat_times": 3,
  "finished": 1,
  "statistics": {
    "median_length": 512.0,
    "mean_length": 551.0,
    "p90_length": 606.4,
    "p95_length": 618.2,
    "max_length": 630.0,
    "min_length": 511.0,
    "std_length": 55.8629,
    "cv_length": 0.1013
  }
}
```

---

### Field Description

| Field | Description |
|------|-------------|
| `sample_id` | Unique identifier generated from prompt-model pair |
| `prompt_id` | Hash-based identifier of the original prompt |
| `prompt_text` | Original user prompt |
| `dataset_name` | Source dataset name |
| `task_type` | Task category: chat, math, or long-sequence |
| `inference_model` | LLM used for generation |
| `output_token_lengths` | Token lengths from repeated independent generations |
| `repeat_times` | Number of repeated generations |
| `finished` | Sampling termination indicator |
| | `1`: early stopping due to length stability |
| | `0`: terminated after reaching maximum sampling rounds |
| `statistics` | Statistical characteristics of output length distribution |

---

### Output Length Statistics

| Statistic | Description |
|-|-|
| `median_length` | Median output length, representing typical generation length |
| `mean_length` | Average output length |
| `p90_length` | 90th percentile output length |
| `p95_length` | 95th percentile output length, representing long-tail generation behavior |
| `max_length` | Maximum observed output length |
| `min_length` | Minimum observed output length |
| `std_length` | Standard deviation of output lengths |
| `cv_length` | Coefficient of variation |

---

### Dataset Source

The MoLD training dataset (`MoLD_shuffled_train.jsonl`) contains approximately 14K samples collected from three datasets: GSM8K, WildChat, and LongBench. The prompts are randomly sampled with seed 42, including 1,200 samples from GSM8K, 2,000 samples from WildChat, and 1,250 samples from LongBench (35 samples are randomly selected from each LongBench task category). Each selected prompt is independently inferred by three instruction-tuned LLMs: Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, and Gemma-3-4B-Instruct. The generated responses from these models are combined to construct the final training set. We also provide `MoLD_sorted_train.jsonl`, where identical prompts are grouped together for data inspection and verification. This sorted version is not recommended for training due to potential data distribution bias.

The MoLD test dataset (`MoLD_shuffled_test.jsonl`) contains approximately 1.8K samples constructed using the same pipeline. Prompts are randomly sampled with seed 46, including 200 samples from GSM8K, 200 samples from WildChat, and 200 samples from LongBench (10 samples are randomly selected from each LongBench task category). Each prompt is also inferred by Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, and Gemma-3-4B-Instruct, and all generated outputs are combined to form the final test set.