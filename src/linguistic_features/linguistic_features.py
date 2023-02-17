import pickle
import os
from src.preprocess import char_index
from src.preprocess.iso_code_to_language import ISOCodeToLanguage


def create_char_indices(train_data, pickle_file_out):
    if not os.path.isfile(pickle_file_out):
        char_index.create_char_indices(train_data, pickle_file_out)

    with open(pickle_file_out, 'rb') as f:
        loaded_char_indices = pickle.load(f, encoding="utf-8")

    print(loaded_char_indices.get_char_index("他"))  # returns 283 for data/train.csv
    print(loaded_char_indices.get_char_bigram_index("シャ"))  # returns 2149 for data/train.csv
    print(loaded_char_indices[0, "unigram"])  # returns 'M' for data/train.csv
    print(loaded_char_indices[0, "bigram"])  # returns '<s>M' for data/train.csv


def make_iso_codes():
    iso_codes = ISOCodeToLanguage()
    return iso_codes
