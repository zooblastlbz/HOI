#!/bin/bash
# 多GPU高性能配置示例
# 适用于大规模数据处理

# 配置参数
INPUT_DATA="/path/to/your/lmdb"           # LMDB 数据目录
OUTPUT_DIR="results_large"                # 输出目录
NUM_SAMPLES=""                            # 处理样本数量（留空则处理全部）

# LLM 配置 - 使用 4 个 GPU 进行张量并行
LLM_MODEL="/path/to/your/llm_model"       # vLLM 模型路径
LLM_TENSOR_PARALLEL_SIZE=4                # LLM tensor parallel size

# SAM 配置 - 使用 2 个 GPU，batch_size=8
SAM_MODEL="/path/to/your/sam3_model"      # SAM3 模型路径
SAM_GPU_IDS="4,5"                         # SAM 使用 GPU 4 和 5
SAM_BATCH_SIZE=8                          # SAM batch size

# YOLO 配置 - 使用 2 个 GPU，batch_size=16
YOLO_MODEL="/path/to/your/yolo_pose.pt"   # YOLO 模型路径
YOLO_GPU_IDS="6,7"                        # YOLO 使用 GPU 6 和 7
YOLO_BATCH_SIZE=16                        # YOLO batch size

# 构建命令
CMD="python pipeline/main_pipeline.py \
    --input ${INPUT_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --llm_model ${LLM_MODEL} \
    --llm_tensor_parallel_size ${LLM_TENSOR_PARALLEL_SIZE} \
    --sam_model ${SAM_MODEL} \
    --sam_gpu_ids ${SAM_GPU_IDS} \
    --sam_batch_size ${SAM_BATCH_SIZE} \
    --yolo_model ${YOLO_MODEL} \
    --yolo_gpu_ids ${YOLO_GPU_IDS} \
    --yolo_batch_size ${YOLO_BATCH_SIZE}"

# 添加可选参数
if [ -n "${NUM_SAMPLES}" ]; then
    CMD="${CMD} --num_samples ${NUM_SAMPLES}"
fi

echo "Running full pipeline with multi-GPU configuration..."
echo "Command: ${CMD}"
eval ${CMD}
