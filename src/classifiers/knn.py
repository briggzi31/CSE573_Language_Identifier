"""
Module Name
-----------
`knn`

Description
-----------
This module provides a script that loads a 2D numpy array from a pickle file,
and loads the true labels from a pickle file, and performs k nearest neighbors
 classification on the vectors. The classification results are returned
as a list of predicted labels.

Usage
-----
"""
import sys
import numpy as np
from src.classifiers.classify import Classifier
from sklearn.neighbors import KNeighborsClassifier

class KNN(Classifier):
    def __init__(self, train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                 test_vectors_file, test_gold_labels_file):
        print('Initializing kNN model...')
        super().__init__(train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                         test_vectors_file, test_gold_labels_file)

        self.model = KNeighborsClassifier(n_neighbors=5)
        return

    def train(self):
        print('Training the kNN model...')
        self.model.fit(self.train_vectors, self.train_gold_labels)
        return

    def test(self, use_dev, output_file):
        print('Testing the kNN model...')
        if use_dev:
            data = self.dev_vectors
            gold_labels = self.dev_gold_labels
        else:
            data = self.test_vectors
            gold_labels = self.test_gold_labels

        predictions = self.model.predict(self.test_vectors)

        confusion_matrix = Classifier.confusion_matrix(predictions, gold_labels)
        classification_report = Classifier.classification_report(predictions, gold_labels)
        acc = Classifier.get_accuracy(gold_labels, predictions)

        with open(output_file, 'w') as output:
            output.write(str(confusion_matrix))
            output.write(classification_report)
            output.write('Accuracy: ' + str(acc))
        return

