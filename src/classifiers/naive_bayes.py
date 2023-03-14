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
# import seaborn as sns
# import matplotlib.pyplot as plt
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
from src.classifiers.classify import Classifier


class NaiveBayes(Classifier):
    def __init__(self, train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                 test_vectors_file, test_gold_labels_file):
        print("Loading data...")
        super().__init__(train_vectors_file, train_gold_labels_file, dev_vectors_file, dev_gold_labels_file,
                         test_vectors_file, test_gold_labels_file)
        self.classes = list(set(self.train_gold_labels))

        self.model = MultinomialNB()
        self.normalized_test = None
        return

    def train(self):
        min_max_scaler = MinMaxScaler()

        # (1) Normalize the data vectors: scales the data between 0 and 1
        print('Normalizing data vectors...')
        # 1. min max scaler
        normalized_train = min_max_scaler.fit_transform(self.train_vectors)
        self.normalized_test = min_max_scaler.fit_transform(self.test_vectors)

        # (2) Split data into batches
        print('Splitting data into batches...')
        train_batches = np.split(normalized_train, 6)
        train_gold_label_batches = np.split(np.array(self.train_gold_labels), 6)

        # (3) Fit Naive Bayes classifier according to X, y.
        print('Fitting the model...')
        for i in tqdm(range(len(train_batches))):
            self.model.partial_fit(train_batches[i], train_gold_label_batches[i], classes=self.classes)
        return

    def predict(self, use_dev, output_file):
        # (4) Predict
        print('Predicting labels for test data...')
        predictions = self.model.predict(self.normalized_test)
        gold_labels = self.test_gold_labels

        confusion_matrix = Classifier.confusion_matrix(predictions, gold_labels)
        classification_report = Classifier.classification_report(predictions, gold_labels)

        with open(output_file, 'w') as output:
            output.write(str(confusion_matrix))
            output.write(classification_report)
        return

    def main(self):
        # Testing with smaller datasets
        # sub_sample = normalized_train[0:1000000, :]
        # y = self.train_gold_labels[0:1000000]
        # xt, xtes, yt, ytes = train_test_split(sub_sample, y, test_size=0.2, random_state=1)
        # model.fit(xt, yt)
        # y_pred = model.predict(xtes)
        # y_true = ytes
        # self.compute_accuracy_score(y_true, y_pred, list(set(y)))
        return

    #     # Heatmap
    #     sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
    #                 xticklabels=classes, yticklabels=classes)
    #     plt.xlabel('true label')
    #     plt.ylabel('predicted label')
    #     plt.show()
    #     return


