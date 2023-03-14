"""
Module Name
-----------
`decision_tree_classification`

Description
-----------
This module provides a script that loads a 2D numpy array from a pickle file,
and loads the true labels from a pickle file, and performs Decision Tree
 classification on the vectors. The classification results are returned
as a list of predicted labels.

Usage
-----
"""

import numpy as np
from src.classifiers.classify import Classifier
from sklearn import tree


class DecisionTree:

    def __init__(self, train_vectors_file, train_gold_labels_file,
                 dev_vectors_file, dev_gold_labels_file,
                 test_vectors_file, test_gold_labels_file, max_depth=20
                 ):

        self.hypermodel = Classifier(train_vectors_file, train_gold_labels_file,
                                     dev_vectors_file, dev_gold_labels_file,
                                     test_vectors_file, test_gold_labels_file
                                     )

        self.train_gold_labels = self.hypermodel.train_gold_labels
        self.dev_gold_labels = self.hypermodel.dev_gold_labels
        self.test_gold_labels = self.hypermodel.test_gold_labels

        self.train_vectors = self.hypermodel.train_vectors
        self.dev_vectors = self.hypermodel.dev_vectors
        self.test_vectors = self.hypermodel.test_vectors

        self.decision_tree = tree.DecisionTreeClassifier(random_state=0, max_depth=max_depth)

    def train(self):
        self.decision_tree.fit(self.train_vectors, np.array(self.train_gold_labels))


    def predict(self, use_dev, output_file):
        if use_dev:
            data = self.dev_vectors
            gold_labels = self.dev_gold_labels
        else:
            data = self.test_vectors
            gold_labels = self.test_gold_labels

        predictions = self.decision_tree.predict(data, gold_labels)

        confusion_matrix = Classifier.confusion_matrix(predictions, gold_labels)
        classification_report = Classifier.classification_report(predictions, gold_labels)

        # TODO: output to a file?

        with open(output_file, 'w') as output:
            output.write(str(confusion_matrix))
            output.write(classification_report)
