#!/bin/bash
# 运行 Step 3: YOLO Pose Estimation

# 配置参数
INPUT_DATA="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/data/mini_dataset_100/dataset.json"                    # 输入数据文件或LMDB目录
OUTPUT_DIR="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/output"                      # 输出目录
NUM_SAMPLES=""          
# YOLO 配置
YOLO_MODEL="/ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/yolo11x-pose.pt"              # YOLO Pose 模型路径
YOLO_GPU_IDS="0,1,2,3,4,5,6,7"                          # YOLO 使用的 GPU IDs (逗号分隔，如 "0,1")
YOLO_BATCH_SIZE=1                         # YOLO batch size

# 构建命令
CMD="python pipeline/main_pipeline.py \
    --input ${INPUT_DATA} \
    --output_dir ${OUTPUT_DIR} \
    --step 3 \
    --yolo_model ${YOLO_MODEL} \
    --yolo_gpu_ids ${YOLO_GPU_IDS} \
    --yolo_batch_size ${YOLO_BATCH_SIZE}"

# 添加可选参数
if [ -n "${NUM_SAMPLES}" ]; then
    CMD="${CMD} --num_samples ${NUM_SAMPLES}"
fi

echo "Running Step 3: Pose Estimation (YOLO)..."
echo "Command: ${CMD}"
eval ${CMD}
