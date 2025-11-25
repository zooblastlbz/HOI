#!/bin/bash
# 运行完整的 HOI Pipeline (所有步骤)

# 配置参数
INPUT_DATA="data.json"                    # 输入数据文件或LMDB目录
OUTPUT_DIR="results"                      # 输出目录
NUM_SAMPLES=""                            # 处理样本数量（留空则处理全部）

# LLM 配置
LLM_MODEL="facebook/opt-125m"             # vLLM 模型路径
LLM_TENSOR_PARALLEL_SIZE=1                # LLM 使用的 GPU 数量

# SAM 配置
SAM_MODEL="facebook/sam3-demo"            # SAM3 模型路径
SAM_GPU_IDS="0"                           # SAM 使用的 GPU IDs (逗号分隔)
SAM_BATCH_SIZE=1                          # SAM batch size

# YOLO 配置
YOLO_MODEL="yolo11n-pose.pt"              # YOLO 模型路径
YOLO_GPU_IDS="0"                          # YOLO 使用的 GPU IDs (逗号分隔)
YOLO_BATCH_SIZE=1                         # YOLO batch size

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

echo "Running full pipeline..."
echo "Command: ${CMD}"
eval ${CMD}
