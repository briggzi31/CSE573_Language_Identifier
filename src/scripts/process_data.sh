#!/bin/sh

python3 process_data.py $1 $2 $3 $4
echo "finished preprocessing training data"

# ex: ./src/scripts/process_data.sh data/train.csv pickle_objects/train_char_indices.pickle pickle_objects/train_topk.pickle
# ex: ./src/scripts/process_data.sh data/test.csv pickle_objects/test_char_indices.pickle pickle_objects/test_topk.pickle


# testing: ./src/scripts/process_data.sh data/testing.csv data/test.csv pickle_objects/testing_char_indices.pickle pickle_objects/testing_topk.pickle


# 1: train_data
# 2: test_data
# 3: indices pickle file
# 4: top_k pickle file