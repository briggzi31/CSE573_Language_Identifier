import sys
from src.linguistic_features.linguistic_features import LinguisticFeatures
from src.preprocess.topk_chars_terms import TopKCharsAndTerms


def process_test_data():
    pass



def process_train_data(train_data, indices_pickle_file, topk_pickle_file):
    indices = LinguisticFeatures.create_char_indices(train_data, indices_pickle_file)
    topk_chars_terms = TopKCharsAndTerms.create_topk_chars_and_terms(topk_pickle_file, indices_pickle_file)

    # print("topk", topk_chars_terms.topk_terms)
    # print("topk counter", topk_chars_terms.term_counts)

    return indices, topk_chars_terms


def create_feature_vectors(train_data, test_data, indices, topk_chars_terms):
    pass
    # train_vectors = LinguisticFeatures(train_data, indices)
    # test_vectors = LinguisticFeatures(test_data, indices)

    # lf = LinguisticFeatures(
    #     "/Users/sambriggs/Documents/CLMS/Winter_2023/CSE_573/final_project/data/train.csv",
    #     "/Users/sambriggs/Documents/CLMS/Winter_2023/CSE_573/final_project/pickle_objects/train_char_indices.pickle")
    # lf.get_linguistic_features()
    # test_text = "तू आज काय करतेयस?"
    # test_word_features = lf.get_word_level_features(test_text, 3)
    # print(test_word_features)



if __name__ == '__main__':
    train_data = sys.argv[1]
    test_data = sys.argv[2]
    indices_pickle_file = sys.argv[3]
    topk_pickle_file = sys.argv[4]

    indices, topk_chars_terms = process_train_data(train_data, indices_pickle_file, topk_pickle_file)

    create_feature_vectors(train_data, test_data, indices, topk_chars_terms)




