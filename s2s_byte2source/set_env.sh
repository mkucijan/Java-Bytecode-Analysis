
export MAIN_DIR=$(dirname "$0")
export MODEL_DIR=${MAIN_DIR}/model
export DATA_PATH=${MAIN_DIR}/db

export VOCAB_SOURCE=${DATA_PATH}/vocab_source.txt
export VOCAB_TARGET=${DATA_PATH}/vocab_target.txt
export TRAIN_SOURCES=${DATA_PATH}/train_sources.txt
export TRAIN_TARGETS=${DATA_PATH}/train_targets.txt
export DEV_SOURCES=${DATA_PATH}/dev_sources.txt
export DEV_TARGETS=${DATA_PATH}/dev_targets.txt

export DEV_TARGETS_REF=${DATA_PATH}/dev_targets.txt
export PRED_DIR=${MODEL_DIR}/pred
export TRAIN_STEPS=1000000
