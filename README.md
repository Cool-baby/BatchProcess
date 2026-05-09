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
