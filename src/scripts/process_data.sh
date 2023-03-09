#!/bin/sh

python3 process_data.py $1 $2 $3 $4 $5 $6 $7 $8
echo "finished preprocessing train and test data, created feature vectors"


# 1: train_data
# 2: test_data
# 3: indices pickle file
# 4: top_k pickle file
# 5: train_vectors pickle file
# 6: train data gold labels pickle file (y true)
# 7: dev set vectors pickle file
# 8: dev set gold labels pickle file (y true)

# testing: ./src/scripts/process_data.sh data/testing.csv data/test.csv pickle_objects/testing_char_indices.pickle pickle_objects/testing_topk.pickle pickle_objects/testing_train_vectors.pickle pickle_objects/testing_gold_labels.pickle

# run: ./src/scripts/process_data.sh data/train.csv data/test.csv pickle_objects/char_indices.pickle pickle_objects/topk.pickle pickle_objects/train_vectors.pickle pickle_objects/gold_labels.pickle
