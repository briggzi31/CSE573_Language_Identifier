import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import f1_score


class Classifier:
    def __init__(self, train_vectors_file, train_gold_labels_file,
                 dev_vectors_file, dev_gold_labels_file,
                 test_vectors_file, test_gold_labels_file):
        # TODO: test vectors and test gold labels
        # feature vectors
        self.train_vectors_file = train_vectors_file
        self.dev_vectors_file = dev_vectors_file
        self.test_vectors_file = test_vectors_file

        # gold labels
        self.train_gold_labels_file = train_gold_labels_file
        self.dev_gold_labels_file = dev_gold_labels_file
        self.test_gold_labels_file = test_gold_labels_file

        self.dev_gold_labels = None
        self.train_gold_labels = None
        self.test_gold_labels = None

        self.dev_vectors = None
        self.train_vectors = None
        self.test_vectors = None

        self.load_data()
        return

    def load_data(self):
        with open(self.train_vectors_file, 'rb') as file:
            self.train_vectors = pickle.load(file)
        with open(self.dev_vectors_file, 'rb') as file:
            self.dev_vectors = pickle.load(file)
        with open(self.test_vectors_file, 'rb') as file:
            self.test_vectors = pickle.load(file)
        with open(self.train_gold_labels_file, 'rb') as file:
            self.train_gold_labels = pickle.load(file)
        with open(self.dev_gold_labels_file, 'rb') as file:
            self.dev_gold_labels = pickle.load(file)
        with open(self.test_gold_labels_file, 'rb') as file:
            self.test_gold_labels = pickle.load(file)
        return

    @ staticmethod
    def confusion_matrix(predictions, gold_labels):
        """
        :param predictions: A ndarray of predicted labels
        :param gold_labels: A ndarray of gold labels

        :return: A string as a confusion matrix, using sklearns confusion_matrix
        Per sklearns documentation:
            Confusion matrix whose i-th row and j-th column entry indicates the number
            of samples with true label being i-th class and predicted label being j-th class.
        """
        matrix_header = "columns are predicted labels, rows are true labels:\n"
        return matrix_header + str(confusion_matrix(gold_labels, predictions)) + "\n"

    @staticmethod
    def classification_report(predictions, gold_labels):
        """
        :param predictions: A ndarray of predicted labels
        :param gold_labels: A ndarray of gold labels

        :return: A string representation of the classification report
        """
        # hides divide by zero warning
        return classification_report(gold_labels, predictions, zero_division=0)

    @staticmethod
    def get_accuracy(y_true, y_pred):
        """
        :param y_pred: A ndarray of predicted labels
        :param y_true: A ndarray of gold labels

        :return: A string representation of the classification report
        """
        acc = accuracy_score(y_true, y_pred)
        return acc


if __name__ == '__main__':
    c = Classifier('pickle_objects/features/train_vectors.pickle',
                    'pickle_objects/gold_labels/train_gold_labels.pickle', 'pickle_objects/features/dev_vectors.pickle',
                    'pickle_objects/gold_labels/dev_gold_labels.pickle', 'pickle_objects/features/test_vectors.pickle',
                   'pickle_objects/gold_labels/test_gold_labels.pickle')
    # print(c.train_vectors[0:2])
    # print(c.train_gold_labels[0:2])
    #
    # print(c.dev_vectors[0:2])
    # print(c.dev_gold_labels[0:2])
    #
    # print(c.test_vectors[0:2])
    # print(c.test_gold_labels[0:2])


