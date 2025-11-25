#!/bin/bash
# 运行 Step 1: LLM 分析

# 配置参数
INPUT_DATA="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/mini_dataset_100/lmdb"                    # 输入数据文件或LMDB目录
OUTPUT_DIR="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/output"                      # 输出目录
NUM_SAMPLES=""                            # 处理样本数量（留空则处理全部）

# LLM 配置
LLM_MODEL="/ytech_m2v5_hdd/workspace/kling_mm/Models/Qwen3-VL-8B-Instruct"             # vLLM 模型路径
LLM_TENSOR_PARALLEL_SIZE=8                # LLM 使用的 GPU 数量 (tensor parallel size)

# 构建命令
CMD="python pipeline/main_pipeline.py \
    --input ${INPUT_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --step 1 \
    --llm_model ${LLM_MODEL} \
    --llm_tensor_parallel_size ${LLM_TENSOR_PARALLEL_SIZE}"

# 添加可选参数
if [ -n "${NUM_SAMPLES}" ]; then
    CMD="${CMD} --num_samples ${NUM_SAMPLES}"
fi

echo "Running Step 1: LLM Analysis..."
echo "Command: ${CMD}"
eval ${CMD}
