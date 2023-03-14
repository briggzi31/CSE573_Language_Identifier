"""
Module Name
-----------
`naive_bayes`

Description
-----------
This module provides a script that loads a 2D numpy array from a pickle file,
and loads the true labels from a pickle file, and performs multinomial Naive
Bayes classification on the vectors. The classification results are returned
as a list of predicted labels.

Usage
-----
"""
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

from classify import Classifier


class NaiveBayes(Classifier):
    def __init__(self, train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                 test_vectors_file, test_gold_labels_file):
        print("Loading data...")
        super().__init__(train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                         test_vectors_file, test_gold_labels_file)
        self.classes = list(set(self.train_gold_labels))
        return

    def main(self):
        model = MultinomialNB()
        min_max_scaler = MinMaxScaler()

        # (1) Normalize the data vectors: scales the data between 0 and 1
        print('Normalizing data vectors...')
        # 1. min max scaler
        normalized_train = min_max_scaler.fit_transform(self.train_vectors)
        normalized_test = min_max_scaler.fit_transform(self.test_vectors)

        # (2) Split data into batches
        print('Splitting data into batches...')
        train_batches = np.split(normalized_train, 6)
        train_gold_label_batches = np.split(np.array(self.train_gold_labels), 6)

        # (3) Fit Naive Bayes classifier according to X, y.
        print('Fitting the model...')
        for i in tqdm(range(len(train_batches))):
            model.partial_fit(train_batches[i], train_gold_label_batches[i], classes=self.classes)

        # (4) Predict
        print('Predicting labels for test data...')
        y_pred = model.predict(normalized_test)
        y_true = self.test_gold_labels

        # (5) Compute confusion matrix
        self.compute_accuracy_score(y_true, y_pred, self.classes)

        # Testing with smaller datasets
        # sub_sample = normalized_train[0:1000000, :]
        # y = self.train_gold_labels[0:1000000]
        # xt, xtes, yt, ytes = train_test_split(sub_sample, y, test_size=0.2, random_state=1)
        # model.fit(xt, yt)
        # y_pred = model.predict(xtes)
        # y_true = ytes
        # self.compute_accuracy_score(y_true, y_pred, list(set(y)))
        return

    def compute_accuracy_score(self, y_true, y_pred, classes):
        # Accuracy score
        acc = accuracy_score(y_true, y_pred)
        print('Accuracy: ', acc)
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        print('Confusion matrix: ')
        print(cm)

        # Heatmap
        sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.show()
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
    nb = NaiveBayes('../../pickle_objects/features/train_vectors.pickle',
                    '../../pickle_objects/gold_labels/train_gold_labels.pickle', '../../pickle_objects/features/dev_vectors.pickle',
                    '../../pickle_objects/gold_labels/dev_gold_labels.pickle', '../../pickle_objects/features/test_vectors.pickle',
                    '../../pickle_objects/gold_labels/test_gold_labels.pickle')
    # nb = NaiveBayes('pickle_objects/features/train_vectors.pickle',
    #                 'pickle_objects/gold_labels/train_gold_labels.pickle', 'pickle_objects/features/dev_vectors.pickle',
    #                 'pickle_objects/gold_labels/dev_gold_labels.pickle', 'pickle_objects/features/test_vectors.pickle',
    #                 'pickle_objects/gold_labels/test_gold_labels.pickle')
    nb.main()
    # print(nb.train_vectors[0:1])

