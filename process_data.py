import pickle
import sys

from src.linguistic_features import linguistic_features
from src.preprocess.char_index import CharIndex
from src.preprocess.iso_code_to_language import ISOCodeToLanguage


if __name__ == '__main__':

    train_data = sys.argv[1]
    pickle_file = sys.argv[2]
    # train_data = "data/test.csv"
    # pickle_file = "pickle_objects/test_char_indices.pickle"

    linguistic_features.create_char_indices(train_data, pickle_file)

    iso_to_lang = linguistic_features.make_iso_codes()
    print(iso_to_lang)
    print(iso_to_lang["eng"])
    # TODO: do we want iso codes in a pickle file as well?


