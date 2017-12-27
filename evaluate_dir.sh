#!/bin/bash -x

DIR=$1
IMAGE_FILE=$(ls $DIR/cropped.jp2)
DATA_FILE=$(ls $DIR/*data)
./evaluate.sh $IMAGE_FILE $DATA_FILE
