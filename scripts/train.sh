#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=/home/ccr/anaconda3/envs/mars3d/bin/python
TRAIN_CODE=train.py

DATASET=semantic_kitti
CONFIG=config
EXP_NAME=debug
WEIGHT=false
RESUME=false


while getopts "p:d:c:n:w:r:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"



EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if ${RESUME}
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=model_last.pth
else
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointseg "$CODE_DIR"
fi
export PYTHONPATH=./$CODE_DIR
echo "Running Code in: $CODE_DIR"


echo " =========> RUN TASK <========="

if ${RESUME}
then
  $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    "$CONFIG_DIR" \
    --options save_path="$EXP_DIR" resume="$MODEL_DIR"/"$WEIGHT"
else
  $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    "$CONFIG_DIR" \
    --options save_path="$EXP_DIR" weight="$WEIGHT"
fi
