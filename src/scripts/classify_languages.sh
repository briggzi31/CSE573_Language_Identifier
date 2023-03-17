#!/bin/sh

python3 classify_languages.py $1 $2 $3 $4 $5 $6
echo "finished classifying languages"

# 1: k_char
# 2: k_word
# 3: k_char_bg
# 4: classifier, either 0, 1, 2, or 3 for baseline, DecisionTree, MultinomialNB, or Logistic Regression respectively
# 5: data for classification, True to use Dev data, False to use Test data
# 6: output file for accuracy

# for testing purposes only:
# ./src/scripts/classify_languages.sh 3 3 3 1 True accuracy/decision_tree/testing_decision_tree

# to run:
# ./src/scripts/classify_languages.sh pickle_objects/features/train_vectors.pickle pickle_objects/gold_labels/train_gold_labels.pickle pickle_objects/features/dev_vectors.pickle pickle_objects/gold_labels/dev_gold_labels.pickle pickle_objects/features/test_vectors.pickle pickle_objects/gold_labels/test_gold_labels.pickle 1 True accuracy/baseline/

