# STABS: Resource-Aware Batch Scheduling with Statistical Token-Length Prediction for LLM Requests in Regional Clouds

---

This repository provides the source code, datasets, and trained model parameters for STABS, a resource-aware batch scheduling method for large-scale LLM inference in regional clouds. STABS addresses the heterogeneity of LLM requests by jointly predicting the median and P95 output token lengths from multiple generations of the same prompt, and exploits prefix-sharing opportunities and complementary GPU resource demands for efficient batch construction. Specifically, STABS integrates statistical token-length prediction, prefix-sharing graph-based spectral clustering, and two-stage bidirectional scanning to improve GPU resource utilization and reduce inference latency.

## Overview

---

### Motivations

#### Motivations for Token Output Length Prediction

Output lengths vary substantially across prompt types and models, particularly for Chat requests. Thus, a single-generation output length may not reliably characterize the output-length behavior of a prompt. Repeated sampling can provide more stable statistical targets, but exhaustively sampling every prompt introduces considerable inference overhead. Our observations show that output lengths of most prompts converge well, suggesting that stable statistical information can often be obtained with only a limited number of samples.

![OutputLengthDistrib.png](https://github.com/Cool-baby/img-storage/blob/main/for_readme/BatchProcess/OutputLengthDistrib.png?raw=true)

#### Motivations for Token Output Length Prediction

Existing prefix trees provide limited flexibility in capturing complex prefix-sharing relationships among requests. Moreover, requests with different computational and memory demands need to be effectively coordinated within each group. This requires accurately characterizing task resource demands from stable output-length estimates while accounting for long-tail requests, so as to better exploit resource complementarity and improve batch efficiency.

### **Key Components**

#### Token Output Length Prediction

![TrainAndPrediction.png](https://github.com/Cool-baby/img-storage/blob/main/for_readme/BatchProcess/TrainAndPrediction.png?raw=true)

STABS predicts the statistical output-length characteristics of each LLM request through two stages:

- **Training Phase:** We first use **Adaptive Prompt-conditioned Sampling (APS)** to dynamically determine the sampling budget for each prompt, collecting multiple generations until the estimated output-length statistics become stable. The resulting statistical targets are used to fine-tune a lightweight **Qwen2.5-1.5B** predictor with LoRA. The predictor jointly learns to estimate the **Median (Med)** and **P95** output lengths using the last-token hidden representation, providing information about both typical and long-tail output demands.

![OutputLengthPredictResult.png](https://github.com/Cool-baby/img-storage/blob/main/for_readme/BatchProcess/OutputLengthPredictResult.png?raw=true)

- **Prediction Phase:** For incoming requests, the trained predictor performs **batch inference with a single forward pass** to simultaneously estimate their Med and P95 output lengths. The predicted statistics are then used as prior information for subsequent batch scheduling.

![P95Result.png](https://github.com/Cool-baby/img-storage/blob/main/for_readme/BatchProcess/P95Result.png?raw=true)

#### Batch Scheduling

![BatchOptimization.png](https://github.com/Cool-baby/img-storage/blob/main/for_readme/BatchProcess/BatchOptimization.png?raw=true)

STABS constructs inference batches by jointly exploiting **prefix-sharing opportunities** and **GPU resource complementarity**. First, a prefix-sharing graph is constructed based on request similarity, and **spectral clustering** groups requests with high prefix similarity to promote KV-cache reuse and reduce redundant computation. Within each group, requests are sorted by computational density, and a **two-stage bidirectional scanning strategy** matches compute-intensive and memory-intensive requests from opposite ends. The strategy first prioritizes resource-complementary combinations under the batch length constraint and then relaxes the constraint when necessary to fill remaining capacity, thereby reducing resource idleness and improving batch efficiency.

## Environment

The experiments are conducted on a Linux server with the following main software environment:

- **Compiler:** GCC 11.2
- **CUDA:** 12.4
- **cuDNN:** 8.9.3.28
- **Python:** 3.11
- **PyTorch:** 2.5.1
- **Torchvision:** 0.20.1
- **Transformers:** 4.52.4
- **Accelerate:** 1.7.0
- **PEFT:** 0.14.0
- **Scikit-learn:** 1.9.0
- **NumPy:** 1.26.4

Before running the code, load the required system modules:

```
gcc/11.2 cuda/12.4 cudnn/8.9.3.28_cuda12.x
```

The complete Python dependencies are provided in `requirements.txt`.

To set up the environment, run the following commands:

```
conda create -n for_batch_proc python==3.12
conda activate for_batch_proc
pip install -r requirements.txt
```

## Hyperparameter Settings

|                Hyperparameter                | Value |
| :------------------------------------------: | :---: |
|          Maximum Input Token Length          | 2048  |
|         Maximum Output Token Length          | 1024  |
|                 Temperature                  |  0.8  |
|                    Top-p                     | 0.95  |
|       Allowed Median Deviation $\tau$        |  5%   |
| Maximum Sampling Times per Prompt $K_{\max}$ |  16   |
| Minimum Sampling Times per Prompt $K_{\min}$ |   3   |

## Run

The token output length predictor model weights are split into two parts due to the large file size. Before running the prediction code, first merge the two model weight files into a single model file:

```
python model_split_tool.py merge core/predictor/STABS_prod_model.part1 core/predictor/STABS_prod_model.part2 core/predictor/STABS_prod_model.pt
```

After merging, update the model weight path in `main` to point to the merged model file:

```
TOKEN_PREDICTOR = {"base_model" : "....../Qwen2.5-1.5B", "checkpoint_path": "/core/predictor/STABS_prod_model.pt"}
```

After configuring the environment and correctly setting the dataset paths, baseline model paths, and other required parameters, run `main.py` to perform batch scheduling experiments:

``````
python main.py
``````

- If only token output length prediction is required, use the prediction methods provided in `core/predictor/token_length_predictor.py`. The `predict_one` method can be used to predict the output length of a single prompt, while `predict_batch` supports batch prediction for multiple prompts.

- To evaluate additional batch scheduling algorithms, the repository adopts a factory design pattern. New schedulers can be implemented by inheriting from `core/scheduler/base_scheduler.py` and then registered in `core/scheduler/schedule_factory.py`. Once registered, the new scheduler can be selected and evaluated through the existing scheduling pipeline.

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

## Citation

To be updated