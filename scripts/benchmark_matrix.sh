#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

BIN="./build/linux-release/bin/pathways"
OUT_DIR="${1:-output/benchmark_matrix}"
mkdir -p "${OUT_DIR}"

echo "=========================================================================="
echo "  Pathways: Automated 1080p & 4K Single-GPU / Dual-GPU Benchmark Matrix"
echo "  Target Output: ${OUT_DIR}"
echo "=========================================================================="

SCENES=(
    "Breakfast Room|scenes/breakfast-room/breakfast_room_extended.glb|8"
    "Glass of Water|scenes/glass-of-water/glass_of_water_extended.glb|8"
    "Kitchen Set|scenes/Kitchen_set/Kitchen_set.usd|8"
    "Damaged Helmet|scenes/DamagedHelmet.glb|8"
)

RESOLUTIONS=(
    "1080p|1920|1080"
    "4K|3840|2160"
)

MODES=(
    "Single-GPU|"
    "Dual-GPU|--mgpu"
)

LOG_FILE="${OUT_DIR}/matrix_run.log"
> "${LOG_FILE}"

RESULTS_CSV="${OUT_DIR}/matrix_results.csv"
echo "Scene,Resolution,GPU_Mode,Avg_Frame_Time_ms,FPS,Throughput_GRays_sec,Primary_ms,Shade_ms,Shadow_ms,Intersect_ms,Tonemap_ms" > "${RESULTS_CSV}"

for scene_entry in "${SCENES[@]}"; do
    IFS="|" read -r SCENE_NAME SCENE_PATH BOUNCES <<< "${scene_entry}"
    echo ""
    echo ">>> Evaluating Scene: ${SCENE_NAME} (Max Bounces: ${BOUNCES}) <<<"

    for res_entry in "${RESOLUTIONS[@]}"; do
        IFS="|" read -r RES_NAME WIDTH HEIGHT <<< "${res_entry}"

        for mode_entry in "${MODES[@]}"; do
            IFS="|" read -r MODE_NAME MODE_FLAG <<< "${mode_entry}"

            RUN_TAG="${SCENE_NAME// /_}_${RES_NAME}_${MODE_NAME//-/_}"
            STATS_JSON="${OUT_DIR}/${RUN_TAG}.json"
            TMP_OUT="${OUT_DIR}/${RUN_TAG}.log"

            printf "  %-16s | %-5s | %-10s ... " "${SCENE_NAME}" "${RES_NAME}" "${MODE_NAME}"

            CMD="${BIN} --headless --scene ${SCENE_PATH} --width ${WIDTH} --height ${HEIGHT} --warmup-frames 15 --frames 60 --spp 1 --max-bounces ${BOUNCES} ${MODE_FLAG} --dump-stats ${STATS_JSON}"
            echo "Running: ${CMD}" >> "${LOG_FILE}"

            if ${CMD} > "${TMP_OUT}" 2>&1; then
                cat "${TMP_OUT}" >> "${LOG_FILE}"
                
                # Extract key metrics from STATS_JSON using jq
                FRAME_TIME=$(jq -r '.performance.avg_frame_time_ms // "N/A"' "${STATS_JSON}")
                FPS=$(jq -r '.performance.avg_fps // "N/A"' "${STATS_JSON}")
                THROUGHPUT=$(jq -r '.performance.gigarays_per_second // "N/A"' "${STATS_JSON}")
                PRIMARY=$(jq -r '.performance.configurations_breakdown[0].pipeline_stages_ms.classify_ms // 0.0' "${STATS_JSON}")
                TONEMAP=$(jq -r '.performance.configurations_breakdown[0].pipeline_stages_ms.tonemap_ms // 0.0' "${STATS_JSON}")

                # Sum shade, shadow, intersect across all bounces
                SHADE_SUM=$(jq -r '[.performance.configurations_breakdown[0].pipeline_stages_ms.bounces[]?.shade_ms] | add // 0.0' "${STATS_JSON}")
                SHADOW_SUM=$(jq -r '[.performance.configurations_breakdown[0].pipeline_stages_ms.bounces[]?.shadow_ms] | add // 0.0' "${STATS_JSON}")
                INTERSECT_SUM=$(jq -r '[.performance.configurations_breakdown[0].pipeline_stages_ms.bounces[]?.intersect_ms] | add // 0.0' "${STATS_JSON}")

                echo "${FRAME_TIME} ms (${FPS} FPS, ${THROUGHPUT} GigaRays/s)"
                echo "${SCENE_NAME},${RES_NAME},${MODE_NAME},${FRAME_TIME},${FPS},${THROUGHPUT},${PRIMARY},${SHADE_SUM},${SHADOW_SUM},${INTERSECT_SUM},${TONEMAP}" >> "${RESULTS_CSV}"
            else
                echo "FAILED (Check ${TMP_OUT})"
            fi
        done
    done
done

echo ""
echo "=========================================================================="
echo "  Benchmark Matrix Complete! Results saved to ${RESULTS_CSV}"
echo "=========================================================================="
