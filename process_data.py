import sys
import os
import pickle
from sklearn.model_selection import train_test_split
from src.linguistic_features.linguistic_features import LinguisticFeatures
from src.preprocess.topk_chars_terms import TopKCharsAndTerms


def process_train_data(train_data, indices_pickle_file, topk_pickle_file):
    indices = LinguisticFeatures.create_char_indices(train_data, indices_pickle_file, topk_pickle_file)
    topk_chars_terms = TopKCharsAndTerms.create_topk_chars_and_terms(topk_pickle_file, indices_pickle_file)

    return indices, topk_chars_terms


def dump_pickle_file(filename, content):
    if not os.path.isfile(filename):
        with open(filename, 'wb') as output:
            pickle.dump(content, output, pickle.HIGHEST_PROTOCOL)
    return


def create_feature_vectors(train_data, test_data, indices, topk_chars_terms, train_pickle_file,
                           train_gold_label_pickle_file, dev_pickle_file, dev_gold_label_pickle_file,
                           test_pickle_file, test_gold_label_pickle_file, k1=3, k2=3, k3=3):

    # create train/dev feature vectors
    lf_train = LinguisticFeatures(train_data, indices, topk_chars_terms, k1, k2, k3)
    vectors = lf_train.feature_vectors
    gold_labels = lf_train.gold_labels

    # Perform train dev split
    x_train, x_dev, y_train, y_dev = train_test_split(vectors, gold_labels, test_size=0.25, random_state=1)

    # create test features vectors
    lf_test = LinguisticFeatures(test_data, indices, topk_chars_terms, k1, k2, k3)
    x_test = lf_test.feature_vectors
    y_test = lf_test.gold_labels

    # Save train set to pickle file
    dump_pickle_file(train_pickle_file, x_train)
    dump_pickle_file(train_gold_label_pickle_file, y_train)
    # Save dev set to pickle file
    dump_pickle_file(dev_pickle_file, x_dev)
    dump_pickle_file(dev_gold_label_pickle_file, y_dev)
    # Save test set to pickle file
    dump_pickle_file(test_pickle_file, x_test)
    dump_pickle_file(test_gold_label_pickle_file, y_test)


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
    # Test set feature vectors and true labels
    test_vector_pickle = sys.argv[9]
    test_gold_label_pickle = sys.argv[10]

    create_feature_vectors(
        train_data, test_data, indices_pickle_file, topk_pickle_file, train_vector_pickle, train_gold_label_pickle,
        dev_vector_pickle, dev_gold_label_pickle, test_vector_pickle, test_gold_label_pickle)
