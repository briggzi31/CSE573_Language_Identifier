import pickle
import sys

from src.linguistic_features.linguistic_features import LinguisticFeatures
# from src.preprocess.char_index import CharIndex, create_char_indices
# from src.preprocess.iso_code_to_language import ISOCodeToLanguage


if __name__ == '__main__':

    train_data = sys.argv[1]
    pickle_file = sys.argv[2]
    # train_data = "data/test.csv"
    # pickle_file = "pickle_objects/test_char_indices.pickle"

    # print(train_data)

    indices = LinguisticFeatures.create_char_indices(train_data, pickle_file)
    iso_to_lang = LinguisticFeatures.make_iso_codes()




    print(iso_to_lang["eng"]) # should return "English"
