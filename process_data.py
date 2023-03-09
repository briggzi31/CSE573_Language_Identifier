import sys
import os
import pickle
from src.linguistic_features.linguistic_features import LinguisticFeatures
from src.preprocess.topk_chars_terms import TopKCharsAndTerms


def process_test_data():
    pass



def process_train_data(train_data, indices_pickle_file, topk_pickle_file):
    indices = LinguisticFeatures.create_char_indices(train_data, indices_pickle_file)
    topk_chars_terms = TopKCharsAndTerms.create_topk_chars_and_terms(topk_pickle_file, indices_pickle_file)

    return indices, topk_chars_terms


def create_feature_vectors(train_data, test_data, indices, topk_chars_terms, train_pickle_file, gold_label_pickle_file):
    lf_train = LinguisticFeatures(train_data, indices, topk_chars_terms)
    train_vectors = lf_train.feature_vectors
    train_gold_labels = lf_train.gold_labels

    if not os.path.isfile(train_pickle_file):
        with open(train_pickle_file, 'wb') as output:
            pickle.dump(train_vectors, output, pickle.HIGHEST_PROTOCOL)

    if not os.path.isfile(gold_label_pickle_file):
        with open(gold_label_pickle_file, 'wb') as output:
            pickle.dump(train_gold_labels, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    indices_pickle_file = sys.argv[3]
    topk_pickle_file = sys.argv[4]
    train_vector_pickle = sys.argv[5]
    gold_label_pickle_file = sys.argv[6]

    create_feature_vectors(
        train_data, test_data, indices_pickle_file, topk_pickle_file, train_vector_pickle, gold_label_pickle_file)
