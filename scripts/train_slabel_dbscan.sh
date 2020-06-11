#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
dropout=$4
name=$5
choice_c=$6

if [ $# -ne 6 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH> <dropout> <name> <choice_c>"
    exit 1
fi


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 MLT_train_dbscan_new.py -tt ${TARGET} -st ${SOURCE} -a ${ARCH}\
	--num-instances 4 --lr 0.00035 --iters 200 -b 64 --epochs 200 \
	--dropout ${dropout} --n-jobs 16 --choice_c ${choice_c} \
	--init-1 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
	--logs-dir logs/dbscan-${SOURCE}TO${TARGET}/${name}

