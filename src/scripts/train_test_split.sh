#!/bin/sh

python3 src/preprocess/train_test_split.py $1 $2 $3
echo "finished train_test_split"

#1: sentences.csv input path
#2: train.csv output path
#3: test.csv output path

# run: ./src/scripts/train_test_split.sh data/sentences.csv data/train.csv data/test.csv