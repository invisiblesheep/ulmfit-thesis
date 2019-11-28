#!/usr/bin/env bash
set -e

# VOCAB_SIZE=30000
VOCAB_SIZE=$1
DATA_DIR=$2
SENTENCE_FILE="${DATA_DIR}/$3"
TEMP_DIR="${DATA_DIR}/processed/up_low/tmp"
OUTPUT_DIR="${DATA_DIR}/processed/up_low/tmp"
SENTENCEPIECE_MODEL_NAME="${OUTPUT_DIR}/sp-${VOCAB_SIZE}"

# sort sentences and remove duplicates
mkdir -p "${TEMP_DIR}"
UNIQ_SENTENCE_FILE="${TEMP_DIR}/train_uniq.txt"
if [ ! -f "${UNIQ_SENTENCE_FILE}" ] ; then
    sort "${SENTENCE_FILE}" | uniq > "${UNIQ_SENTENCE_FILE}"
fi

# train sentencepiece model
if [ ! -f "${SENTENCEPIECE_MODEL_NAME}.model" ]; then
  python ./train_spm.py --sentence_file "${UNIQ_SENTENCE_FILE}" --sentencepiece_model_name "${SENTENCEPIECE_MODEL_NAME}" --vocab_size ${VOCAB_SIZE}
else
  echo "Setencepiece model '${SENTENCEPIECE_MODEL_NAME}.model' already exists."
fi
