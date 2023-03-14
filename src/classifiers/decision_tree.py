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
import pandas as pd
from classify import Classifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,classification_report


class DecisionTree:

    def __init__(self):
        # self.hypermodel = Classifier()
        pass

    def load_data(self):
        pass