#!/bin/bash
# 运行 Step 2: SAM3 Visual Grounding

# 配置参数
INPUT_DATA="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/mini_dataset_100/dataset.json"                    # 输入数据文件或LMDB目录
OUTPUT_DIR="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/output"                      # 输出目录
NUM_SAMPLES=""                            # 处理样本数量（留空则处理全部）

# SAM 配置
SAM_MODEL="/ytech_m2v5_hdd/workspace/kling_mm/Models/sam3/"            # SAM3 模型路径或 HuggingFace ID
SAM_GPU_IDS="0,1,2,3,4,5,6,7"                           # SAM 使用的 GPU IDs (逗号分隔，如 "0,1,2")
SAM_BATCH_SIZE=1                          # SAM batch size

# 构建命令
CMD="python pipeline/main_pipeline.py \
    --input ${INPUT_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --step 2 \
    --sam_model ${SAM_MODEL} \
    --sam_gpu_ids ${SAM_GPU_IDS} \
    --sam_batch_size ${SAM_BATCH_SIZE}"

# 添加可选参数
if [ -n "${NUM_SAMPLES}" ]; then
    CMD="${CMD} --num_samples ${NUM_SAMPLES}"
fi

echo "Running Step 2: Visual Grounding (SAM3)..."
echo "Command: ${CMD}"
eval ${CMD}
