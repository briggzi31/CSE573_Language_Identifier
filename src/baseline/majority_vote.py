"""
Module Name
-----------
`majority_vote_classification`

Description
-----------
This module provides a script that loads a 2D numpy array from a pickle file,
and loads the true labels from a pickle file, and performs majority vote
 classification on the vectors. The classification results are returned
as a list of predicted labels.

Usage
-----
"""

import numpy as np
from src.classifiers.classify import Classifier
from statistics import mode


class MajorityVote(Classifier):

    def __init__(self, train_vectors_file, train_gold_labels_file,
                 dev_vectors_file, dev_gold_labels_file,
                 test_vectors_file, test_gold_labels_file,
                 ):

        super().__init__(train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                         test_vectors_file, test_gold_labels_file)

        self.most_common = None


    def train(self):
        self.most_common = mode(self.train_gold_labels)


    def predict(self, use_dev, output_file):
        if use_dev:
            gold_labels = self.dev_gold_labels
        else:
            gold_labels = self.test_gold_labels

        predictions = np.array([self.most_common for i in range(len(gold_labels))])

        confusion_matrix = Classifier.confusion_matrix(predictions, gold_labels)
        classification_report = Classifier.classification_report(predictions, gold_labels)

        with open(output_file, 'w') as output:
            output.write(str(confusion_matrix))
            output.write(classification_report)
