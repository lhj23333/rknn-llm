#!/usr/bin/env bash
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS_DIR="${BASE_DIR}/../../models"
RKNN_ENV="${BASE_DIR}/../../rknn_env"
RKLLM_ENV="${BASE_DIR}/../../rkllm_env"

DATASET_JSON="${BASE_DIR}/data/datasets.json"

INPUT_DIR="${BASE_DIR}/data"
OUTPUT_DIR="${BASE_DIR}/rkllm"
LOG_DIR="${BASE_DIR}/logs"

mkdir -p "${INPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "BASE_DIR   : ${BASE_DIR}"
echo "MODELS_DIR : ${MODELS_DIR}"
echo "RKNN_ENV   : ${RKNN_ENV}"
echo "RKLLM_ENV  : ${RKLLM_ENV}"
echo "=========================================="
echo

run_one () {

MODEL_NAME="$1"
MODEL_PATH="$2"
MODEL_TYPE="$3"
INPUT_JSON="$4"
OUTPUT_RKLLM="$5"

echo "=========================================="
echo "MODEL : ${MODEL_NAME}"
echo "PATH  : ${MODEL_PATH}"
echo "=========================================="

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] model path not found: ${MODEL_PATH}"
  exit 1
fi

############################################
# STEP 1 生成量化输入集
############################################

echo
echo "[STEP 1] Generate quant dataset"
echo "activate rknn_env"

source "${RKNN_ENV}/bin/activate"

python data/make_input_embeds_for_quantize.py \
  --path "${MODEL_PATH}" \
  --model_name "${MODEL_TYPE}" \
  --dataset_json "${DATASET_JSON}" \
  --output_json "${INPUT_JSON}" \
  2>&1 | tee "${LOG_DIR}/${MODEL_NAME}_quant.log"

deactivate

if [ ! -f "${INPUT_JSON}" ]; then
  echo "[ERROR] quant dataset failed: ${INPUT_JSON}"
  exit 1
fi

############################################
# STEP 2 导出 RKLLM
############################################

echo
echo "[STEP 2] Export RKLLM"
echo "activate rkllm_env"

source "${RKLLM_ENV}/bin/activate"

python export/export_rkllm.py \
  --path "${MODEL_PATH}" \
  --target-platform rk3588 \
  --num_npu_core 1 \
  --quantized_dtype w8a8 \
  --device cpu \
  --dataset "${INPUT_JSON}" \
  --savepath "${OUTPUT_RKLLM}" \
  2>&1 | tee "${LOG_DIR}/${MODEL_NAME}_rkllm.log"

deactivate

if [ ! -f "${OUTPUT_RKLLM}" ]; then
  echo "[ERROR] rkllm export failed: ${OUTPUT_RKLLM}"
  exit 1
fi

echo
echo "[DONE] ${MODEL_NAME}"
echo

}

############################################
# Qwen3-VL-4B
############################################

run_one \
"Qwen3-VL-4B" \
"${MODELS_DIR}/Qwen3-VL-4B" \
"qwen3-vl" \
"${INPUT_DIR}/inputs_qwen3-vl-4B.json" \
"${OUTPUT_DIR}/Qwen3-VL-4B.rkllm"

############################################
# Internvl_3.5_1B
############################################

run_one \
"Internvl_3.5_1B" \
"${MODELS_DIR}/Internvl_3.5_1B" \
"internvl3.5" \
"${INPUT_DIR}/inputs_internvl3.5-1B.json" \
"${OUTPUT_DIR}/Intervl3.5-1B.rkllm"

############################################
# Internvl_3.5_2B
############################################

run_one \
"Internvl_3.5_2B" \
"${MODELS_DIR}/Internvl_3.5_2B" \
"internvl3.5" \
"${INPUT_DIR}/inputs_internvl3.5-2B.json" \
"${OUTPUT_DIR}/Intervl3.5-2B.rkllm"

############################################
# Internvl_3.5_4B
############################################

run_one \
"Internvl_3.5_4B" \
"${MODELS_DIR}/Internvl_3.5_4B" \
"internvl3.5" \
"${INPUT_DIR}/inputs_internvl3.5-4B.json" \
"${OUTPUT_DIR}/Intervl3.5-4B.rkllm"

echo
echo "=========================================="
echo "ALL MODELS FINISHED"
echo "=========================================="