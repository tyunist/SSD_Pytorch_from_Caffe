#!/bin/bash

# Install required packages
python3.7 -m pip install --user -r docs_n_setups/package_requirements.txt 

INPUT_CHANNELS=3
NUM_CLASSES=19

TRAIN_MODEL_DEF_CONFIG='configs/model_configs/ASU_SSD_model/three_drones/train_SSD300.prototxt'
DEPLOY_MODEL_DEF_CONFIG='configs/model_configs/ASU_SSD_model/three_drones/deploy_SSD300.prototxt'
PRETRAINED_CKPT='pretrained_ckpt/best_checkpoint.pth.tar'
DATA_CONFIG_PREFIX='configs/data_configs'

# TODO: change the enviroment variables according to your machine, if you are running on host machine.
# For running on docker, can skip this step! 
# 1. Set the home directory that contain code, data directories

HOMEDIR="."
# 2. Set the media directory that contains log, models, results
LOGDIR="meta_data"


# 3. Set dataloader (data used for training + testing
# By default, using TN_penstock
# All data sets are stored in HOMEDIR/data
DATASET="three_drones"

# 4. Set image size
# TODO: raw image's size
INPUT_H="512"
INPUT_W="640"

# Image input to the network
CROP_IMG_SIZE=300

if [ -n "$1" ]
then 
  echo "Given a specific data set: $1"
  DATASET=$1
else
  echo "NO specific data set is given, by default, use $DATASET"
fi 


# Default image size of three drone experiments (at Ian's house)
if [ $DATASET == "three_drones" ]
then
  INPUT_H=540
  INPUT_W=960
  INPUT_CHANNELS=3
  NUM_CLASSES=4
  echo "By default, image size to $INPUT_H (H) x $INPUT_W (W)" 
  echo "By default, num of classes is $NUM_CLASSES" 
  echo "By default, num of input channels is $INPUT_CHANNELS"
fi

if [ $DATASET == "all_drones" ]
then
  INPUT_H=540
  INPUT_W=960
  INPUT_CHANNELS=3
  NUM_CLASSES=19
  echo "By default, image size to $INPUT_H (H) x $INPUT_W (W)" 
  echo "By default, num of classes is $NUM_CLASSES" 
  echo "By default, num of input channels is $INPUT_CHANNELS"
fi



if [ -n "$2" ]
then 
  echo "Given a specific image height: $2"
  INPUT_H=$2
else
  echo "NO specific image height is given, by default, use ${INPUT_H}"
fi 

if [ -n "$3" ]
then 
  echo "Given a specific image width: $3"
  INPUT_W=$3
else
  echo "NO specific image width is given, by default, use ${INPUT_W}"
fi 


# Execute variable setting  
export HOMEDIR=${HOMEDIR}
export LOGDIR=${LOGDIR}
export INPUT_H=${INPUT_H}
export INPUT_W=${INPUT_W}
export CROP_IMG_SIZE=${CROP_IMG_SIZE}
export DATASET=${DATASET}
export NUM_CLASSES=${NUM_CLASSES}
export INPUT_CHANNELS=${INPUT_CHANNELS}
export TRAIN_MODEL_DEF=${HOMEDIR}/${TRAIN_MODEL_DEF_CONFIG}
export DEPLOY_MODEL_DEF=${HOMEDIR}/${DEPLOY_MODEL_DEF_CONFIG}
export DATA_CONFIG=${HOMEDIR}/${DATA_CONFIG_PREFIX}/${DATASET}.data
export PRETRAINED_CKPT=${HOMEDIR}/${PRETRAINED_CKPT}


echo "Set DATASET as ${DATASET}"
echo "Set HOMEDIR as ${HOMEDIR}"
echo "Set LOGDIR as ${LOGDIR}"
echo "Set INPUT_H as ${INPUT_H}"
echo "Set INPUT_W as ${INPUT_W}"
echo "Set CROP_IMG_SIZE as ${CROP_IMG_SIZE}"
echo "Set NUM_CLASSES as ${NUM_CLASSES}"
echo "Set INPUT_CHANNELS as ${INPUT_CHANNELS}"
echo "Set TRAIN_MODEL_DEF as ${TRAIN_MODEL_DEF}"
echo "Set DEPLOY_MODEL_DEF as ${DEPLOY_MODEL_DEF}"
echo "Set PRETRAINED_CKPT as ${PRETRAINED_CKPT}"
echo "Set DATA_CONFIG as ${DATA_CONFIG}"

exec $SHELL -i

