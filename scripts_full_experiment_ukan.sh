#!/bin/bash
# AT-UKanNet 完整实验脚本
# 训练 AT-UKanNet（UKAN + ATConv + 并行 CASA）在指定数据集上的表现，
# 支持不同的 attention 变体（默认推荐 parallel，对应并行 CASA）
# 支持7个数据集并行训练，GPU并发管理

DATA_DIR="./inputs"
# OUTPUT_DIR会在脚本中设置，这里只是占位符
EPOCHS=400
BATCH_SIZE=8
LR=1e-4
K_FOLDS=5

# 数据集配置现在在参数设置部分根据DEFAULT_DATASETS动态生成


# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ${SCRIPT_DIR}

# 设置输出目录
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# ==================== 默认参数设置 ====================
# 默认数据集：根据需要修改
DEFAULT_DATASETS="bus_bra,TN3K,cvc,ours,busi_whu,busi,glas"

# 默认GPU列表：1,3
DEFAULT_GPU_LIST="3"
# 默认GPU并发数配置：GPU3最多2个（格式：GPU_ID:MAX_COUNT）
DEFAULT_MAX_CONCURRENT_PER_GPU_LIST="3:1"
# 默认随机种子数量：3个
DEFAULT_NUM_SEEDS=3
# 默认要运行的折数：all 表示运行所有折，也可以指定 "0,1" 表示只运行第0折和第1折
DEFAULT_FOLD_TO_RUN="all"
# 默认attention变体：parallel (可选: serial, multiscale, parallel, global, hybrid, pyramid)
DEFAULT_ATTENTION_VARIANT="parallel"

# ==================== 参数设置 ====================
# GPU配置
GPU_LIST_STR="${DEFAULT_GPU_LIST}"
MAX_CONCURRENT_PER_GPU="${MAX_CONCURRENT_PER_GPU}"
# 解析按GPU的并发数配置
declare -A GPU_MAX_CONCURRENT
for gpu_config in ${DEFAULT_MAX_CONCURRENT_PER_GPU_LIST}; do
    IFS=':' read -r gpu_id max_count <<< "${gpu_config}"
    GPU_MAX_CONCURRENT[$gpu_id]=$max_count
done

# ==================== 数据集映射配置 ====================
# 完整的数据集映射（所有可用数据集）
declare -A dataset_input_size_map=(
    ["busi"]="256"
    ["glas"]="512"
    ["cvc"]="256"
    ["ours"]="256"
    ["busi_whu"]="256"
    ["TN3K"]="256"
    ["bus_bar"]="256"
)

declare -A dataset_seed1_map=(
    ["busi"]="2981"
    ["glas"]="6142"
    ["cvc"]="1187"
    ["ours"]="2224"
    ["busi_whu"]="1234"
    ["TN3K"]="5678"
    ["bus_bar"]="9012"
)

declare -A dataset_seed2_map=(
    ["busi"]="3981"
    ["glas"]="7142"
    ["cvc"]="2187"
    ["ours"]="3224"
    ["busi_whu"]="2234"
    ["TN3K"]="6678"
    ["bus_bar"]="10012"
)

declare -A dataset_seed3_map=(
    ["busi"]="4981"
    ["glas"]="8142"
    ["cvc"]="3187"
    ["ours"]="4224"
    ["busi_whu"]="3234"
    ["TN3K"]="7678"
    ["bus_bar"]="11012"
)

# 根据DEFAULT_DATASETS筛选数据集
datasets=()
input_size=()
dataseed1=()
dataseed2=()
dataseed3=()

IFS=',' read -ra SELECTED_DATASETS <<< "${DEFAULT_DATASETS}"
for ds in "${SELECTED_DATASETS[@]}"; do
    # 去除空白字符
    ds=$(echo "$ds" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ -n "${dataset_input_size_map[$ds]}" ]; then
        datasets+=("$ds")
        input_size+=("${dataset_input_size_map[$ds]}")
        dataseed1+=("${dataset_seed1_map[$ds]}")
        dataseed2+=("${dataset_seed2_map[$ds]}")
        dataseed3+=("${dataset_seed3_map[$ds]}")
    else
        echo "警告: 数据集 '$ds' 未在映射中找到，跳过"
    fi
done

NUM_SEEDS=${DEFAULT_NUM_SEEDS}
FOLD_TO_RUN="${DEFAULT_FOLD_TO_RUN}"
ATTENTION_VARIANT="${DEFAULT_ATTENTION_VARIANT}"

# 转换GPU列表字符串为数组
IFS=',' read -ra GPU_LIST <<< "${GPU_LIST_STR}"

# ==================== 模型配置 ====================
# AT-UKanNet 模型配置
MODEL_NAME="AT_UKanNet"
ARCH="AT-UKanNet"
USE_ATCONV="True"
ATCONV_ENCODER_LAYERS="3"
ATCONV_DECODER_LAYERS="1"
EDGE_LOSS_WEIGHT="0.15"
LOSS="BCEDiceLoss"

# ==================== 创建日志目录 ====================
LOG_DIR="${SCRIPT_DIR}/logs/UKAN_full_experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}

# ==================== 函数定义 ====================
# 函数：获取指定GPU的最大并发数限制
get_gpu_max_concurrent() {
    local gpu=$1
    if [ -n "${GPU_MAX_CONCURRENT[$gpu]}" ]; then
        echo ${GPU_MAX_CONCURRENT[$gpu]}
    else
        echo ${MAX_CONCURRENT_PER_GPU}
    fi
}

# ==================== 打印配置信息 ====================
echo "=========================================="
echo "UKAN+ATConv+Attention_Variant 完整实验"
echo "=========================================="
echo "模型: ${MODEL_NAME} (${ARCH})"
echo "Attention变体: ${ATTENTION_VARIANT}"
echo "数据集: ${datasets[@]} (共 ${#datasets[@]} 个)"
echo "每个数据集: ${NUM_SEEDS}个随机种子 × ${K_FOLDS}折交叉验证"
# 计算实际运行的折数
if [ "${FOLD_TO_RUN}" == "all" ]; then
    ACTUAL_FOLDS=${K_FOLDS}
    FOLD_INFO="${K_FOLDS}折（全部）"
else
    ACTUAL_FOLDS=$(echo "${FOLD_TO_RUN}" | tr ',' '\n' | wc -l)
    FOLD_INFO="折 ${FOLD_TO_RUN}（共 ${ACTUAL_FOLDS} 折）"
fi
echo "运行的折数: ${FOLD_INFO}"
echo "总实验数: $((${#datasets[@]} * ${NUM_SEEDS} * ${ACTUAL_FOLDS}))"
echo "使用GPU: ${GPU_LIST[@]}"
if [ ${#GPU_MAX_CONCURRENT[@]} -gt 0 ]; then
    echo "每张GPU最多并发:"
    for gpu in "${GPU_LIST[@]}"; do
        max_concurrent=$(get_gpu_max_concurrent $gpu)
        echo "  GPU ${gpu}: ${max_concurrent}"
    done
else
    echo "每张GPU最多并发: ${MAX_CONCURRENT_PER_GPU}"
fi
echo "训练轮数: ${EPOCHS}"
echo "批次大小: ${BATCH_SIZE}"
echo "学习率: ${LR}"
echo "日志目录: ${LOG_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# ==================== 存储所有后台进程的PID和对应的GPU ====================
declare -a pids=()
declare -a gpu_ids=()
declare -a exp_datasets=()
declare -a exp_seeds=()
declare -a exp_sizes=()

# ==================== 函数定义 ====================

# 函数：运行单个实验
run_experiment() {
    local dataset=$1
    local dataseed=$2
    local input_size=$3
    local gpu_id=$4
    local log_file=$5

    # 设置实验名称
    local exp_name="${MODEL_NAME}_${dataset}_seed${dataseed}"

    echo "[GPU ${gpu_id}] 开始实验: ${MODEL_NAME} ${dataset} (seed: ${dataseed}, size: ${input_size})" | tee -a ${log_file}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验开始" | tee -a ${log_file}

    # 构建Python命令（UKAN+ATConv+Attention_Variant）
    local python_cmd="CUDA_VISIBLE_DEVICES=${gpu_id} bash -c 'source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ATUKANet && python ${SCRIPT_DIR}/train_atconv_pfan.py \
        --name \"${exp_name}\" \
        --datasets ${dataset} \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --arch ${ARCH} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --input_w ${input_size} \
        --input_h ${input_size} \
        --k_folds ${K_FOLDS} \
        --fold_to_run ${FOLD_TO_RUN} \
        --dataseed ${dataseed} \
        --gpu_ids ${gpu_id} \
        --use_atconv ${USE_ATCONV} \
        --atconv_encoder_layers ${ATCONV_ENCODER_LAYERS} \
        --atconv_decoder_layers ${ATCONV_DECODER_LAYERS} \
        --use_attention True \
        --attention_variant ${ATTENTION_VARIANT} \
        --use_hybrid_arch False '"

    eval ${python_cmd} >> ${log_file} 2>&1

    local exit_code=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验结束" | tee -a ${log_file}
    if [ $exit_code -eq 0 ]; then
        echo "[GPU ${gpu_id}] ✓ 实验 ${dataset} (seed: ${dataseed}) 完成" | tee -a ${log_file}
    else
        echo "[GPU ${gpu_id}] ✗ 实验 ${dataset} (seed: ${dataseed}) 失败 (退出码: ${exit_code})" | tee -a ${log_file}
    fi
    return $exit_code
}

# 函数：获取指定GPU上当前运行的进程数
get_gpu_running_count() {
    local gpu=$1
    local count=0
    for i in "${!gpu_ids[@]}"; do
        if [ "${gpu_ids[$i]}" == "$gpu" ]; then
            if kill -0 ${pids[$i]} 2>/dev/null; then
                ((count++))
            fi
        fi
    done
    echo $count
}

# 函数：清理已完成的进程
cleanup_finished_processes() {
    local new_pids=()
    local new_gpu_ids=()
    local new_datasets=()
    local new_seeds=()
    local new_sizes=()

    for i in "${!pids[@]}"; do
        if kill -0 ${pids[$i]} 2>/dev/null; then
            new_pids+=(${pids[$i]})
            new_gpu_ids+=(${gpu_ids[$i]})
            new_datasets+=(${exp_datasets[$i]})
            new_seeds+=(${exp_seeds[$i]})
            new_sizes+=(${exp_sizes[$i]})
        else
            wait ${pids[$i]}
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "✓ 一个实验已完成，释放GPU ${gpu_ids[$i]}的槽位"
            else
                echo "✗ 一个实验失败，释放GPU ${gpu_ids[$i]}的槽位"
            fi
        fi
    done

    pids=("${new_pids[@]}")
    gpu_ids=("${new_gpu_ids[@]}")
    exp_datasets=("${new_datasets[@]}")
    exp_seeds=("${new_seeds[@]}")
    exp_sizes=("${new_sizes[@]}")
}

# 函数：分配GPU（选择当前运行进程最少的GPU）
allocate_gpu() {
    local min_count=999
    local selected_gpu=""

    for gpu in "${GPU_LIST[@]}"; do
        local count=$(get_gpu_running_count $gpu)
        local max_concurrent=$(get_gpu_max_concurrent $gpu)

        if [ $count -lt $max_concurrent ]; then
            if [ $count -lt $min_count ]; then
                min_count=$count
                selected_gpu=$gpu
            fi
        fi
    done

    echo ${selected_gpu}
}


# ==================== 计算实际运行的折数 ====================
if [ "${FOLD_TO_RUN}" == "all" ]; then
    ACTUAL_FOLDS=${K_FOLDS}
else
    # 计算逗号分隔的折数数量
    ACTUAL_FOLDS=$(echo "${FOLD_TO_RUN}" | tr ',' '\n' | wc -l)
fi

# ==================== 生成所有实验配置 ====================
echo "生成实验配置..."
total_experiments=0
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    size=${input_size[$i]}

    seeds=()
    if [ $NUM_SEEDS -ge 1 ]; then
        seeds+=(${dataseed1[$i]})
    fi
    if [ $NUM_SEEDS -ge 2 ]; then
        seeds+=(${dataseed2[$i]})
    fi
    if [ $NUM_SEEDS -ge 3 ]; then
        seeds+=(${dataseed3[$i]})
    fi

    for seed in "${seeds[@]}"; do
        ((total_experiments++))
    done
done

if [ "${FOLD_TO_RUN}" == "all" ]; then
    echo "总共 ${total_experiments} 个实验（每个实验${K_FOLDS}折交叉验证）"
else
    echo "总共 ${total_experiments} 个实验（每个实验运行折 ${FOLD_TO_RUN}，共 ${ACTUAL_FOLDS} 折）"
fi
echo "开始并行训练..."
echo ""

# ==================== 并行运行所有实验 ====================
experiment_idx=0

echo ""
echo "=========================================="
echo "开始运行模型: ${MODEL_NAME}"
echo "=========================================="
echo ""

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    size=${input_size[$i]}

    seeds=()
    if [ $NUM_SEEDS -ge 1 ]; then
        seeds+=(${dataseed1[$i]})
    fi
    if [ $NUM_SEEDS -ge 2 ]; then
        seeds+=(${dataseed2[$i]})
    fi
    if [ $NUM_SEEDS -ge 3 ]; then
        seeds+=(${dataseed3[$i]})
    fi

    for seed in "${seeds[@]}"; do
        cleanup_finished_processes

        while true; do
            cleanup_finished_processes
            gpu=$(allocate_gpu)
            if [ "$gpu" != "" ]; then
                count=$(get_gpu_running_count $gpu)
                max_concurrent=$(get_gpu_max_concurrent $gpu)
                if [ $count -lt $max_concurrent ]; then
                    break
                fi
            fi
            sleep 5
        done

        LR_FOR_FILENAME=$(echo "${LR}" | sed 's/\./_/g' | sed 's/-/_/g')
        log_file="${LOG_DIR}/${MODEL_NAME}_${dataset}_seed${seed}_kfold${K_FOLDS}_gpu${gpu}_batch${BATCH_SIZE}_lr${LR_FOR_FILENAME}.log"

        gpu_count=$(get_gpu_running_count $gpu)
        gpu_max=$(get_gpu_max_concurrent $gpu)
        echo "[$((experiment_idx+1))/${total_experiments}] 启动实验 ${MODEL_NAME} ${dataset} (seed: ${seed}, size: ${size}) 在 GPU ${gpu} (GPU ${gpu} 当前并发: ${gpu_count}/${gpu_max})"
        run_experiment "${dataset}" "${seed}" "${size}" "${gpu}" "${log_file}" &

        pids+=($!)
        gpu_ids+=($gpu)
        exp_datasets+=(${dataset})
        exp_seeds+=(${seed})
        exp_sizes+=(${size})
        ((experiment_idx++))

        sleep 2
    done
done

echo ""
echo "=========================================="
echo "所有实验已启动，等待完成..."
echo "总实验数: ${total_experiments}"
echo "进程ID: ${pids[@]}"
echo "=========================================="
echo ""

# ==================== 等待所有后台进程完成 ====================
failed_count=0
declare -a failed_experiments=()
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    gpu=${gpu_ids[$i]}
    dataset=${exp_datasets[$i]}
    seed=${exp_seeds[$i]}
    size=${exp_sizes[$i]}

    echo "等待实验 (PID: ${pid}, GPU: ${gpu}): ${MODEL_NAME} ${dataset} (seed: ${seed})..."
    if wait $pid; then
        echo "✓ 实验 ${MODEL_NAME} ${dataset} (seed: ${seed}) 成功完成"
    else
        echo "✗ 实验 ${MODEL_NAME} ${dataset} (seed: ${seed}) 失败 (GPU: ${gpu})"
        ((failed_count++))
        failed_experiments+=("${MODEL_NAME}|${dataset}|${seed}")
    fi
done

echo ""
echo "=========================================="
echo "实验完成！"
echo "成功: $((${total_experiments} - failed_count))"
echo "失败: ${failed_count}"
echo "=========================================="

# 如果有失败的实验，直接报错退出
if [ ${#failed_experiments[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "错误：以下实验失败，脚本退出"
    echo "=========================================="
    for failed_exp in "${failed_experiments[@]}"; do
        IFS='|' read -r model dataset seed <<< "${failed_exp}"
        echo "  - ${model} ${dataset} (seed: ${seed})"
    done
    echo ""
    echo "请检查日志文件以获取详细错误信息"
    exit 1
fi

echo ""
echo "=========================================="
echo "所有实验成功完成！"
echo "模型: ${MODEL_NAME} (${ATTENTION_VARIANT})"
echo "日志目录: ${LOG_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
echo ""
echo "正在收集实验结果并生成Excel表格..."
echo ""

# ==================== 运行结果收集脚本 ====================
for dataset in "${datasets[@]}"; do
    echo "收集 ${dataset} 数据集的结果..."
    python ${SCRIPT_DIR}/collect_results_ukan.py \
        --output_dir ${OUTPUT_DIR} \
        --dataset ${dataset} \
        --model_name_prefix ${MODEL_NAME} \
        --excel_file ${OUTPUT_DIR}/full_experiment_${MODEL_NAME}_${dataset}_$(date +%Y%m%d_%H%M%S).csv
done

echo ""
echo "结果已保存到: ${OUTPUT_DIR}/full_experiment_${MODEL_NAME}_*.csv"
