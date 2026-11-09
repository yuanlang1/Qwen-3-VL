#!/bin/bash

set -uo pipefail

# ============================================================
# Usage
# ============================================================

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage:"
    echo "  bash scripts/sportsqa-3b-inference/benchmark_batch_workers.sh MODEL BENCH_NAME [ADAPTER_PATH]"
    exit 2
fi

MODEL=$1
BENCH_NAME=$2
ADAPTER_PATH=${3:-}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

RUNNER="${SCRIPT_DIR}/infer_sportsqa_3b.sh"

# ============================================================
# Benchmark configuration
# ============================================================

# 每组测试多少个样本
SAMPLE_NUM=${SPORTSQA_BENCH_SAMPLES:-960}

# 你的 infer_sportsqa_3b.sh 当前输出目录
OUTPUT_ROOT=${SPORTSQA_OUTPUT_ROOT:-/home/user/data/sportsqa_output/qwen2.5-vl}

# benchmark 日志和结果存放位置
BENCH_ROOT=${SPORTSQA_BENCH_ROOT:-/home/user/data/sportsqa_benchmark/${BENCH_NAME}}

mkdir -p "${BENCH_ROOT}"

RESULT_FILE="${BENCH_ROOT}/results.tsv"

# ============================================================
# Search space
# ============================================================

# 第一阶段：
# 固定 batch，搜索 num_workers
#
# 如果 batch=4 太小，可以之后改成 8/16。
WORKER_SEARCH_BATCH=${SPORTSQA_WORKER_SEARCH_BATCH:-4}

# 默认搜索 workers:
# 1 2 4 6 8
WORKERS_LIST=${SPORTSQA_WORKERS_LIST:-"1 2 4 6 8"}

# 第二阶段：
# 固定最佳 workers，搜索 batch
BATCH_LIST=${SPORTSQA_BATCH_LIST:-"2 4 8 16 24 32 40 48 64"}

# GPU
GPU_INDEX=${SPORTSQA_GPU_INDEX:-0}


# ============================================================
# Check runner
# ============================================================

if [ ! -f "${RUNNER}" ]; then
    echo "Runner not found:"
    echo "${RUNNER}"
    exit 2
fi

if [ ! -e "${MODEL}" ]; then
    echo "Model not found:"
    echo "${MODEL}"
    exit 2
fi


# ============================================================
# Print config
# ============================================================

echo
echo "============================================================"
echo "Sports-QA Benchmark"
echo "============================================================"
echo "model               : ${MODEL}"
echo "benchmark name      : ${BENCH_NAME}"
echo "samples/test        : ${SAMPLE_NUM}"
echo "worker search batch : ${WORKER_SEARCH_BATCH}"
echo "workers             : ${WORKERS_LIST}"
echo "batches             : ${BATCH_LIST}"
echo "output root         : ${OUTPUT_ROOT}"
echo "benchmark root      : ${BENCH_ROOT}"
echo "============================================================"
echo


# ============================================================
# Result header
# ============================================================

printf \
"stage\tbatch\tworkers\tstatus\telapsed_s\tqa_count\twall_qa_s\tavg_gpu_util\tpeak_vram_mib\tpeak_ram_gib\n" \
> "${RESULT_FILE}"


# ============================================================
# Read cgroup memory
# ============================================================

read_ram_bytes() {

    # cgroup v2
    if [ -f /sys/fs/cgroup/memory.current ]; then
        cat /sys/fs/cgroup/memory.current
        return
    fi

    # cgroup v1
    if [ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
        cat /sys/fs/cgroup/memory/memory.usage_in_bytes
        return
    fi

    echo 0
}


# ============================================================
# Resource monitor
#
# 每秒记录：
# GPU utilization
# VRAM
# container RAM
# ============================================================

monitor_resources() {

    local pid=$1
    local output=$2

    echo "timestamp,gpu_util,vram_mib,ram_bytes" > "${output}"

    while kill -0 "${pid}" 2>/dev/null; do

        gpu_line=$(
            nvidia-smi \
                -i "${GPU_INDEX}" \
                --query-gpu=utilization.gpu,memory.used \
                --format=csv,noheader,nounits \
                2>/dev/null \
                | head -1
        )

        if [ -n "${gpu_line}" ]; then

            gpu_util=$(
                echo "${gpu_line}" |
                cut -d',' -f1 |
                xargs
            )

            vram=$(
                echo "${gpu_line}" |
                cut -d',' -f2 |
                xargs
            )

        else
            gpu_util=0
            vram=0
        fi

        ram=$(read_ram_bytes)

        echo \
            "$(date +%s),${gpu_util},${vram},${ram}" \
            >> "${output}"

        sleep 1
    done
}


# ============================================================
# Run one benchmark
# ============================================================

run_test() {

    local stage=$1
    local batch=$2
    local workers=$3

    local run_name="${BENCH_NAME}-${stage}-b${batch}-w${workers}"

    local log_file="${BENCH_ROOT}/${run_name}.log"
    local monitor_file="${BENCH_ROOT}/${run_name}.monitor.csv"

    local run_output="${OUTPUT_ROOT}/${run_name}"

    echo
    echo "============================================================"
    echo "TEST: ${run_name}"
    echo "============================================================"
    echo "stage   : ${stage}"
    echo "batch   : ${batch}"
    echo "workers : ${workers}"
    echo "samples : ${SAMPLE_NUM}"
    echo "============================================================"

    # --------------------------------------------------------
    # 非常重要：
    #
    # infer_sportsqa_3b.sh 使用了 --resume。
    # 如果之前已经跑过同名实验，不删除会导致 benchmark 失真。
    # --------------------------------------------------------

    if [ -d "${run_output}" ]; then
        echo "Removing old benchmark output:"
        echo "${run_output}"
        rm -rf "${run_output}"
    fi


    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    start_time=$(date +%s.%N)

    SPORTSQA_SAMPLE_NUM="${SAMPLE_NUM}" \
    SPORTSQA_BATCH_SIZE="${batch}" \
    SPORTSQA_NUM_WORKERS="${workers}" \
    bash "${RUNNER}" \
        "${MODEL}" \
        "${run_name}" \
        "${ADAPTER_PATH}" \
        > "${log_file}" 2>&1 &

    infer_pid=$!


    # --------------------------------------------------------
    # Monitor
    # --------------------------------------------------------

    monitor_resources \
        "${infer_pid}" \
        "${monitor_file}" &

    monitor_pid=$!


    # --------------------------------------------------------
    # Wait inference
    # --------------------------------------------------------

    wait "${infer_pid}"
    exit_code=$?

    # monitor 会在 infer_pid 消失以后自动退出
    wait "${monitor_pid}" 2>/dev/null || true


    end_time=$(date +%s.%N)

    elapsed=$(
        awk \
            -v start="${start_time}" \
            -v end="${end_time}" \
            'BEGIN {
                printf "%.2f", end-start
            }'
    )


    # ========================================================
    # Determine status
    # ========================================================

    if [ "${exit_code}" -eq 0 ]; then

        status="OK"

    elif grep -Eqi \
        "CUDA out of memory|torch.OutOfMemoryError|CUDA error: out of memory|out of memory" \
        "${log_file}"; then

        status="OOM"

    else

        status="FAIL"

    fi


    # ========================================================
    # Find predictions
    # ========================================================

    prediction_file=$(
        find "${run_output}/predictions" \
            -type f \
            -name "*.jsonl" \
            2>/dev/null \
            | head -1
    )


    if [ -n "${prediction_file}" ] &&
       [ -f "${prediction_file}" ]; then

        qa_count=$(
            wc -l < "${prediction_file}" |
            xargs
        )

    else
        qa_count=0
    fi


    # ========================================================
    # Throughput
    # ========================================================

    wall_qa_s=$(
        awk \
            -v n="${qa_count}" \
            -v t="${elapsed}" \
            'BEGIN {
                if (t > 0) {
                    printf "%.4f", n/t
                } else {
                    printf "0"
                }
            }'
    )


    # ========================================================
    # Average GPU utilization
    # ========================================================

    avg_gpu=$(
        awk -F',' '
            NR > 1 {
                sum += $2
                count++
            }

            END {
                if (count > 0) {
                    printf "%.1f", sum/count
                } else {
                    printf "0"
                }
            }
        ' "${monitor_file}"
    )


    # ========================================================
    # Peak VRAM
    # ========================================================

    peak_vram=$(
        awk -F',' '
            NR > 1 {
                if ($3 > max) {
                    max=$3
                }
            }

            END {
                printf "%.0f", max+0
            }
        ' "${monitor_file}"
    )


    # ========================================================
    # Peak RAM
    # ========================================================

    peak_ram=$(
        awk -F',' '
            NR > 1 {
                if ($4 > max) {
                    max=$4
                }
            }

            END {
                printf "%.2f", max/1024/1024/1024
            }
        ' "${monitor_file}"
    )


    # ========================================================
    # Save
    # ========================================================

    printf \
"%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${stage}" \
        "${batch}" \
        "${workers}" \
        "${status}" \
        "${elapsed}" \
        "${qa_count}" \
        "${wall_qa_s}" \
        "${avg_gpu}" \
        "${peak_vram}" \
        "${peak_ram}" \
        >> "${RESULT_FILE}"


    # ========================================================
    # Print
    # ========================================================

    echo
    echo "---------------- RESULT ----------------"
    echo "name            : ${run_name}"
    echo "status          : ${status}"
    echo "elapsed         : ${elapsed}s"
    echo "QA count        : ${qa_count}"
    echo "wall QA/s       : ${wall_qa_s}"
    echo "avg GPU util    : ${avg_gpu}%"
    echo "peak VRAM       : ${peak_vram} MiB"
    echo "peak cgroup RAM : ${peak_ram} GiB"
    echo "log             : ${log_file}"
    echo "----------------------------------------"


    # --------------------------------------------------------
    # 出错时输出最后一些日志，方便直接看到原因
    # --------------------------------------------------------

    if [ "${status}" != "OK" ]; then

        echo
        echo "Last 30 log lines:"
        echo "----------------------------------------"

        tail -30 "${log_file}" || true

        echo "----------------------------------------"

    fi
}


# ============================================================
# STAGE 1
#
# Search num_workers
# ============================================================

echo
echo
echo "############################################################"
echo "# STAGE 1: worker search"
echo "############################################################"

for workers in ${WORKERS_LIST}; do

    run_test \
        "worker" \
        "${WORKER_SEARCH_BATCH}" \
        "${workers}"

done


# ============================================================
# Find best num_workers
# ============================================================

best_workers=$(
    awk -F'\t' '

        NR > 1 &&
        $1 == "worker" &&
        $4 == "OK" {

            speed=$7+0

            if (speed > best_speed) {
                best_speed=speed
                best_workers=$3
            }
        }

        END {
            if (best_workers != "") {
                print best_workers
            }
        }

    ' "${RESULT_FILE}"
)


if [ -z "${best_workers}" ]; then

    echo
    echo "ERROR:"
    echo "No successful configuration in worker search."
    echo
    echo "Check logs:"
    echo "${BENCH_ROOT}"

    exit 1
fi


echo
echo "============================================================"
echo "STAGE 1 RESULT"
echo "Best num_workers = ${best_workers}"
echo "============================================================"


# ============================================================
# STAGE 2
#
# Search batch size
# ============================================================

echo
echo
echo "############################################################"
echo "# STAGE 2: batch search"
echo "############################################################"

for batch in ${BATCH_LIST}; do

    run_test \
        "batch" \
        "${batch}" \
        "${best_workers}"


    # --------------------------------------------------------
    # 如果已经 OOM，则通常没有必要继续测试更大 batch
    # --------------------------------------------------------

    last_status=$(
        tail -1 "${RESULT_FILE}" |
        cut -f4
    )

    if [ "${last_status}" = "OOM" ]; then

        echo
        echo "============================================================"
        echo "OOM at batch=${batch}"
        echo "Stop searching larger batch sizes."
        echo "============================================================"

        break
    fi

done


# ============================================================
# Find best batch
# ============================================================

best_line=$(
    awk -F'\t' '

        NR > 1 &&
        $1 == "batch" &&
        $4 == "OK" {

            speed=$7+0

            if (speed > best_speed) {
                best_speed=speed
                best_line=$0
            }
        }

        END {
            print best_line
        }

    ' "${RESULT_FILE}"
)


# ============================================================
# Final results
# ============================================================

echo
echo
echo "############################################################"
echo "# ALL RESULTS"
echo "############################################################"
echo

column -t -s $'\t' "${RESULT_FILE}" 2>/dev/null \
    || cat "${RESULT_FILE}"


echo
echo
echo "############################################################"
echo "# SORTED BY WALL QA/S"
echo "############################################################"
echo

{
    head -1 "${RESULT_FILE}"

    tail -n +2 "${RESULT_FILE}" \
        | awk -F'\t' '$4 == "OK"' \
        | sort \
            -t $'\t' \
            -k7,7nr

} | column -t -s $'\t' 2>/dev/null \
  || true


# ============================================================
# Best configuration
# ============================================================

echo
echo
echo "############################################################"
echo "# BEST CONFIGURATION"
echo "############################################################"
echo


if [ -n "${best_line}" ]; then

    best_batch=$(
        echo "${best_line}" |
        cut -f2
    )

    final_workers=$(
        echo "${best_line}" |
        cut -f3
    )

    best_speed=$(
        echo "${best_line}" |
        cut -f7
    )

    best_gpu=$(
        echo "${best_line}" |
        cut -f8
    )

    best_vram=$(
        echo "${best_line}" |
        cut -f9
    )

    best_ram=$(
        echo "${best_line}" |
        cut -f10
    )


    echo "batch_size      = ${best_batch}"
    echo "num_workers     = ${final_workers}"
    echo "wall QA/s       = ${best_speed}"
    echo "avg GPU util    = ${best_gpu}%"
    echo "peak VRAM       = ${best_vram} MiB"
    echo "peak RAM        = ${best_ram} GiB"

    echo
    echo "Recommended runner settings:"
    echo
    echo "batch_size=${best_batch}"
    echo "num_workers=${final_workers}"

else

    echo "No successful batch configuration found."

fi


echo
echo "Full results:"
echo "${RESULT_FILE}"

echo
echo "Logs:"
echo "${BENCH_ROOT}"
