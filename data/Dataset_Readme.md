# MoLD: Multi-model Output Length Distribution Dataset

## Overview

MoLD (**Multi-model Output Length Distribution Dataset**) is a benchmark dataset designed for studying **robust LLM output token length prediction**.

Existing output length prediction methods usually formulate the task as a deterministic regression problem, where each prompt is associated with only one observed output length. However, the generation process of large language models (LLMs) is inherently stochastic, and the same prompt may produce outputs with significantly different lengths across independent generations. Moreover, different LLMs may exhibit distinct output length behaviors for the same input prompt.

To address these challenges, MoLD constructs a **prompt-conditioned output length distribution dataset** by repeatedly querying the same prompt on the same LLM and recording the generated token lengths. Instead of assigning a single output length label, each sample contains an empirical output length distribution together with robust statistical descriptors, enabling research on distribution-aware and uncertainty-aware output length prediction.

---

# Dataset Statistics

MoLD contains repeated inference results from multiple representative open-source LLMs across diverse tasks.

| Category | Description |
|----------|-------------|
| Number of prompts | Approximately 10K |
| Number of LLMs | 3 |
| Task categories | Chat, Math, Long-sequence |
| Source datasets | WildChat, GSM8K, LongBench |
| Inference models | Gemma-3-4B, Llama-3.2-3B-Instruct, Qwen2.5-3B |
| Annotation type | Prompt-conditioned output length distribution |
| Prediction targets | Median output length and P95 output length |

---

# Dataset Composition

Each data sample corresponds to:

> One unique prompt executed on one specific LLM with multiple independent generations.

The dataset does **not** merge the output length distributions of the same prompt across different models, because different LLMs may have different generation preferences and response length characteristics.

The basic sample unit is:

\[
(prompt\_id, inference\_model)
\]

For example, the same prompt may generate three independent length distributions:

| prompt_id | model | Output Length Distribution |
|-----------|-------|----------------------------|
| xxx | Qwen2.5-3B | [120, 135, 128, ...] |
| xxx | Llama-3.2-3B | [80, 95, 90, ...] |
| xxx | Gemma-3-4B | [160, 180, 170, ...] |

These samples share the same prompt but preserve model-specific output behaviors.

---

# Data Format

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

  "output_token_lengths": [
    630,
    512,
    511
  ],

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

# Field Description

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

# Output Length Statistics

For each prompt-model pair, MoLD provides statistical information describing the empirical output length distribution.

The statistics include:

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

Among these statistics, the primary prediction targets are:

1. **Typical Length**
   
\[
L_{median}
\]

which represents the normal expected generation length.

2. **Worst-case Length**

\[
L_{p95}
\]

which captures potential long-tail generation requirements.

---

# Dataset Construction Procedure

The dataset is constructed through the following pipeline:

```
Original Prompt Dataset

        |
        v

Repeated inference on the same LLM

        |
        v

Collect generated token lengths

        |
        v

Estimate prompt-conditioned length distribution

        |
        v

Extract robust statistics

        |
        v

Construct MoLD samples
```

For each prompt:

1. Select an inference model.
2. Perform multiple independent generations.
3. Record generated token lengths.
4. Compute distribution statistics.
5. Store the prompt-model pair as one dataset sample.

---

# Intended Applications

MoLD is designed for research on:

## 1. Robust Output Length Prediction

MoLD enables models to predict:

\[
f(prompt, model)
\rightarrow
(L_{median}, L_{p95})
\]

instead of predicting only a single deterministic output length.

Potential applications include:

- LLM inference scheduling
- GPU memory allocation
- KV-cache management
- Dynamic batching
- Latency-aware serving
- Edge LLM deployment

---

## 2. Distribution-aware Length Estimation

The dataset supports research on:

- Prompt-conditioned length distribution modeling
- Uncertainty-aware prediction
- Quantile regression
- Distribution prediction
- Long-tail output estimation

---

## 3. Cross-model Generalization

Since the same prompts are evaluated on multiple LLMs, MoLD enables studies on:

- Model-aware output length prediction
- Transferability across LLMs
- Model-specific generation behaviors

---

# Recommended Data Split

To avoid information leakage, we recommend splitting the dataset according to `prompt_id` rather than `sample_id`.

Example:

```
Training:

prompt A - Qwen
prompt A - Llama
prompt A - Gemma


Testing:

prompt B - Qwen
prompt B - Llama
prompt B - Gemma
```

This ensures that evaluation measures generalization to unseen prompts.

---

[//]: # (# Citation)

[//]: # ()
[//]: # (If you use MoLD in your research, please cite:)

[//]: # ()
[//]: # (```)

[//]: # (@dataset{mold2026,)

[//]: # (  title={MoLD: Multi-model Output Length Distribution Dataset for Robust LLM Token Length Prediction},)

[//]: # (  year={2026})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (# License)

[//]: # ()
[//]: # (The dataset follows the licenses of the original source datasets:)

[//]: # ()
[//]: # (- WildChat)

[//]: # (- GSM8K)

[//]: # (- LongBench)

[//]: # ()
[//]: # (Users should comply with the corresponding dataset licenses and terms of use.)