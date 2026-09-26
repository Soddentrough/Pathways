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
                
                # Extract key metrics from stdout
                FRAME_TIME=$(grep -m 1 "Average Frame Time:" "${TMP_OUT}" | awk '{print $4}' || echo "N/A")
                FPS=$(grep -m 1 "Average Frame Time:" "${TMP_OUT}" | awk -F'(' '{print $2}' | awk '{print $1}' || echo "N/A")
                THROUGHPUT=$(grep -m 1 "Ray Throughput:" "${TMP_OUT}" | awk '{print $3}' || echo "N/A")
                PRIMARY=$(grep -m 1 "Classify (Primary RayGen):" "${TMP_OUT}" | awk '{print $4}' || echo "0.0")
                TONEMAP=$(grep -m 1 "Tonemap / Resolve:" "${TMP_OUT}" | awk '{print $4}' || echo "0.0")

                # Sum shade, shadow, intersect across all bounces
                SHADE_SUM=$(awk '/- Bounce [0-9]+:.*Shade:/ { for (i=1; i<=NF; i++) if ($i=="Shade:") print $(i+1) }' "${TMP_OUT}" | awk '{s+=$1} END {printf "%.3f", s}')
                SHADOW_SUM=$(awk '/- Bounce [0-9]+:.*Shadow:/ { for (i=1; i<=NF; i++) if ($i=="Shadow:") print $(i+1) }' "${TMP_OUT}" | awk '{s+=$1} END {printf "%.3f", s}')
                INTERSECT_SUM=$(awk '/- Bounce [0-9]+:.*Intersect:/ { for (i=1; i<=NF; i++) if ($i=="Intersect:") print $(i+1) }' "${TMP_OUT}" | awk '{s+=$1} END {printf "%.3f", s}')

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
