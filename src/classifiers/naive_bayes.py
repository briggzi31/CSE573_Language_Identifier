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
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
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

        conf_matrix = Classifier.confusion_matrix(predictions, gold_labels)
        class_report = Classifier.classification_report(predictions, gold_labels)
        acc = Classifier.get_accuracy(gold_labels, predictions)

        with open(output_file, 'w') as output:
            output.write(str(conf_matrix))
            output.write(class_report)
            output.write('Accuracy: ' + str(acc))
        return



