import sys
import os
import pickle
from sklearn.model_selection import train_test_split
from src.linguistic_features.linguistic_features import LinguisticFeatures
from src.preprocess.topk_chars_terms import TopKCharsAndTerms


def process_test_data():
    pass


def process_train_data(train_data, indices_pickle_file, topk_pickle_file):
    indices = LinguisticFeatures.create_char_indices(train_data, indices_pickle_file, topk_pickle_file)
    topk_chars_terms = TopKCharsAndTerms.create_topk_chars_and_terms(topk_pickle_file, indices_pickle_file)

    return indices, topk_chars_terms


def create_feature_vectors(train_data, test_data, indices, topk_chars_terms, train_pickle_file,
                           train_gold_label_pickle_file, dev_pickle_file, dev_gold_labbel_pickle_file):
    lf_train = LinguisticFeatures(train_data, indices, topk_chars_terms)
    vectors = lf_train.feature_vectors
    gold_labels = lf_train.gold_labels

    # Perform train dev split
    x_train, x_dev, y_train, y_dev = train_test_split(vectors, gold_labels)

    # Save train set to pickle file
    if not os.path.isfile(train_pickle_file):
        with open(train_pickle_file, 'wb') as output:
            pickle.dump(x_train, output, pickle.HIGHEST_PROTOCOL)

    if not os.path.isfile(train_gold_label_pickle_file):
        with open(train_gold_label_pickle_file, 'wb') as output:
            pickle.dump(y_train, output, pickle.HIGHEST_PROTOCOL)

    # Save dev set to pickle file
    if not os.path.isfile(dev_pickle_file):
        with open(dev_pickle_file, 'wb') as output:
            pickle.dump(x_dev, output, pickle.HIGHEST_PROTOCOL)

    if not os.path.isfile(dev_gold_labbel_pickle_file):
        with open(dev_gold_labbel_pickle_file, 'wb') as output:
            pickle.dump(y_dev, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    indices_pickle_file = sys.argv[3]
    topk_pickle_file = sys.argv[4]
    # Train set feature vectors and true labels
    train_vector_pickle = sys.argv[5]
    train_gold_label_pickle = sys.argv[6]
    # Dev set feature vectors and true labels
    dev_vector_pickle = sys.argv[7]
    dev_gold_label_pickle = sys.argv[8]

    create_feature_vectors(
        train_data, test_data, indices_pickle_file, topk_pickle_file, train_vector_pickle, train_gold_label_pickle,
        dev_vector_pickle, dev_gold_label_pickle)
