#!/bin/sh
SOURCE=$1
TARGET=$2
ARCH=$3
CLUSTER=$4
name=$5
CLASSES=$6
choice_c=$7

if [ $# -ne 7 ]
  then
    echo "Arguments error: <SOURCE> <TARGET> <ARCH> <CLUSTER> <name> <CLASSES> <choice_c>"
    exit 1
fi

for i in $(seq 0 10)
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python3 MLT_train_single.py -dt ${TARGET} -st ${SOURCE} -a ${ARCH} --num-clusters ${CLUSTER} \
        --num-instances 4 --lr 0.00035 --iters 200 -b 64 --epochs 40 \
        --dropout 0.5 --n-jobs 16\
        --init-1 logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-4/model_best.pth.tar \
        --pesudolabel-path logs/${SOURCE}TO${TARGET}/${ARCH}-selflabel-${name} \
        --logs-dir logs/${SOURCE}TO${TARGET}/${ARCH}-selflabel-${name} \
        --cluster-num ${i} --choice_c ${choice_c} --ncs ${CLASSES}
done

