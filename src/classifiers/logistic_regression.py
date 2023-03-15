"""
Module Name
-----------
`Logistic_Regression_Classification`

Description
-----------
This module provides a script that loads a 2D numpy array from a pickle file,
and loads the true labels from a pickle file, and performs logistic regression
 classification on the vectors. The classification results are returned
as a list of predicted labels.

Usage
-----
"""

import numpy as np
from src.classifiers.classify import Classifier
from sklearn.linear_model import SGDClassifier

class LogisticRegressionClassifier(Classifier):

    def __init__(self, train_vectors_file, train_gold_labels_file,
                 dev_vectors_file, dev_gold_labels_file,
                 test_vectors_file, test_gold_labels_file,
                 ):
        super().__init__(train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                         test_vectors_file, test_gold_labels_file)

        self.model = SGDClassifier(loss='log', penalty='l1', max_iter=10,
                                   learning_rate='optimal', early_stopping=True, n_jobs=4,
                                   )


    def train(self):
        self.model.fit(self.train_vectors, self.train_gold_labels)


    def predict(self, use_dev, output_file):
        if use_dev:
            data = self.dev_vectors
            gold_labels = self.dev_gold_labels
        else:
            data = self.test_vectors
            gold_labels = self.test_gold_labels

        predictions = self.model.predict(data)

        confusion_matrix = Classifier.confusion_matrix(predictions, gold_labels)
        classification_report = Classifier.classification_report(predictions, gold_labels)

        with open(output_file, 'w') as output:
            output.write(str(confusion_matrix))
            output.write(classification_report)
