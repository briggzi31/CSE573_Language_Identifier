#!/bin/sh

# this will run the baseline majority vote for all combinations of topk

./src/scripts/classify_languages.sh 3 3 3 2 True accuracy/nb/acc_3_3_3
echo "finished 3 3 3 (1/8)"

./src/scripts/classify_languages.sh 3 3 9 2 True accuracy/nb/acc_3_3_9
echo "finished 3 3 9 (2/8)"

./src/scripts/classify_languages.sh 3 9 3 2 True accuracy/nb/acc_3_9_3
echo "finished 3 3 9 (3/8)"

./src/scripts/classify_languages.sh 3 9 9 2 True accuracy/nb/acc_3_9_9
echo "finished 3 9 9 (4/8)"

./src/scripts/classify_languages.sh 9 3 3 2 True accuracy/nb/acc_9_3_3
echo "finished 9 3 3 (5/8)"

./src/scripts/classify_languages.sh 9 3 9 2 True accuracy/nb/acc_9_3_9
echo "finished 9 3 9 (6/8)"

./src/scripts/classify_languages.sh 9 9 3 2 True accuracy/nb/acc_9_9_3
echo "finished 9 9 3 (7/8)"

./src/scripts/classify_languages.sh 9 9 9 2 True accuracy/nb/acc_9_9_9
echo "finished 9 9 9 (8/8)"