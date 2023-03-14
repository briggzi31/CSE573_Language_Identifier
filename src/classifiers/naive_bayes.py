"""
Module Name
-----------
`naive_bayes_classification`

Description
-----------
This module provides a script that loads a 2D numpy array from a pickle file,
and loads the true labels from a pickle file, and performs multinomial Naive
Bayes classification on the vectors. The classification results are returned
as a list of predicted labels.

Usage
-----
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from classify import Classifier


class NaiveBayes(Classifier):
    def __init__(self, train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                 test_vectors_file, test_gold_labels_file):
        super().__init__(train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                         test_vectors_file, test_gold_labels_file)
        self.classes = list(set(self.train_gold_labels))
        return

    def main(self):
        model = MultinomialNB()

        # Normalize the data vectors: scales the data between 0 and 1
        print('Normalizing data vectors...')
        min_max_scaler = MinMaxScaler()
        normalized_train = min_max_scaler.fit_transform(self.train_vectors)
        normalized_test = min_max_scaler.fit_transform(self.test_vectors)
        # print(normalized_train[0:2])
        # print(self.train_vectors[0:2])

        # Fit Naive Bayes classifier according to X, y.
        print('Fitting the model...')
        model.fit(normalized_train, self.train_gold_labels)
        # p.fit(self.train_vectors, self.train_gold_labels)

        # Predict
        print('Predicting labels for test data...')
        y_pred = model.predict(normalized_test[0:20])
        y_true = self.test_gold_labels[0:20]
        # p.predict(self.test_vectors[0:2])
        # print(self.test_gold_labels[0:2])

        # Print confusion matrix
        self.compute_accuracy_score(y_true, y_pred)
        return

    def compute_accuracy_score(self, y_true, y_pred):
        # Accuracy score
        acc = accuracy_score(y_true, y_pred)
        print('Accuracy: ', acc)
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        print('Confusion matrix: ')
        print(cm)
        return


if __name__ == '__main__':
    # train_vectors_file = sys.argv[1]
    # train_gold_label_file = sys.argv[2]
    # dev_vectors_file = sys.argv[3]
    # dev_gold_label_file = sys.argv[4]
    # test_vectors_file = sys.argv[5]
    # test_gold_label_file = sys.argv[6]
    #
    # nb = NaiveBayes(train_vectors_file, train_gold_label_file, dev_vectors_file, dev_gold_label_file, test_vectors_file,
    #                 test_gold_label_file)
    # nb.main()

    # Local
    nb = NaiveBayes('pickle_objects/features/train_vectors.pickle',
                    'pickle_objects/gold_labels/train_gold_labels.pickle', 'pickle_objects/features/dev_vectors.pickle',
                    'pickle_objects/gold_labels/dev_gold_labels.pickle', 'pickle_objects/features/test_vectors.pickle',
                    'pickle_objects/gold_labels/test_gold_labels.pickle')
    nb.main()
    # print(nb.train_vectors[0:1])

