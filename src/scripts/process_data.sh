#!/bin/sh

python3 process_data.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}
echo "finished preprocessing train, dev, and test data, created feature vectors"


# 1: train_data
# 2: test_data
# 3: indices pickle file
# 4: top_k pickle file
# 5: train_vectors pickle file
# 6: train data gold labels pickle file (y true)
# 7: dev set vectors pickle file
# 8: dev set gold labels pickle file (y true)
# 9: test set vectors pickle file
# 10: test set gold labels pickle file (y true)

# testing:
# ./src/scripts/process_data.sh data/testing.csv data/test.csv pickle_objects/testing/testing_char_indices.pickle pickle_objects/testing/testing_topk.pickle pickle_objects/testing/features/testing_train_vectors.pickle pickle_objects/testing/gold_labels/testing_train_gold_labels.pickle pickle_objects/testing/features/testing_dev_vectors.pickle pickle_objects/testing/gold_labels/testing_dev_gold_labels.pickle pickle_objects/testing/features/testing_test_vectors.pickle pickle_objects/testing/gold_labels/testing_test_gold_labels.pickle

# run:
# ./src/scripts/process_data.sh data/train.csv data/test.csv pickle_objects/char_indices/char_indices.pickle pickle_objects/topk/topk.pickle pickle_objects/features/train_vectors.pickle pickle_objects/gold_labels/train_gold_labels.pickle pickle_objects/features/dev_vectors.pickle pickle_objects/gold_labels/dev_gold_labels.pickle pickle_objects/features/test_vectors.pickle pickle_objects/gold_labels/test_gold_labels.pickle
