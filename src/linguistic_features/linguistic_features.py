import pickle
import os
import csv
from collections import Counter
from src.preprocess import char_index
from src.preprocess.iso_code_to_language import ISOCodeToLanguage


class LinguisticFeatures:

    def __init__(self, train_data, pickle_file_out):
        self.train_data = train_data
        self.pickle_file_out = pickle_file_out

        self.loaded_char_indices = None
        return

    def get_linguistic_features(self):
        """
        Iterate through the dataset and compute linguistic feature vectors for each document in data.
        """
        # Load char-to-index mapping from pickle file
        self.loaded_char_indices = self.create_char_indices(self.train_data, self.pickle_file_out)

        # Process each instance/doc in train data
        with open(self.train_data, 'r') as file:
            csv_reader = csv.reader(file)
            i = 0
            for doc in csv_reader:
                if i > 0:
                    self.get_char_level_features(doc, 3)
                    self.get_word_level_features()
                    break
                i += 1
        return

    def get_word_level_features(self):
        """
        Computes all linguistic features at word-level.
            Linguistic features computed:
                (1) Top k frequent words
                (2) Average length of the full word
        """
        pass

    def get_char_level_features(self, doc, k=3):
        """
        Computes all linguistic features at character-level.
            Linguistic features computed:
                (1) Top k frequent characters
                (2) Ratio of whitespace to number of chars in doc
                (3) Ratio of alphanumeric chars to other chars  # TODO
                (4) Ratio of lower-case to upper-case chars  # TODO
        """
        chars = [c for c in doc[3]]  # Ignore whitespace
        char_indices = [self.loaded_char_indices.get_char_index(c) for c in chars]

        whitespace_idx = self.loaded_char_indices.get_char_index(' ')  # Index of whitespace

        # Find top k frequent characters
        top_k_char_indices = self.get_top_k_chars(char_indices, whitespace_idx, k)
        print("Top k char indices: ", top_k_char_indices)

        # Calculate ratio of whitespace to number of chars
        chars_count = Counter(char_indices)
        whitespace_ratio = chars_count[whitespace_idx] / sum(chars_count.values())
        print("Ratio of whitespace to num of all chars: ", whitespace_ratio)

        # Calculate ratio of alphanumeric chars to other chars
        # TODO

        # Calculate ratio of lower-case to upper-case chars
        # TODO: To be discussed

        return

    @staticmethod
    def get_top_k_chars(char_indices, whitespace_idx, k=3):
        """
        Find the n most frequent characters in a document (excluding whitespace), and return the indices of the n most
        frequent characters in a list.

        :param doc: list; each doc is one row in the train or test data, where the item at index 2 is ISO code of the
                    language, and the item at index 3 is the text of the document
        :return: list; in which each item is the index of a most frequent char in doc
        """
        char_indices_without_space = [i for i in char_indices if i is not whitespace_idx]

        # TODO: Raise error if doc length < k
        top_n_chars = Counter(char_indices_without_space).most_common(k)
        top_n_char_indices = [char for char, count in top_n_chars]

        return top_n_char_indices

    @staticmethod
    def create_char_indices(train_data, pickle_file_out):
        if not os.path.isfile(pickle_file_out):
            char_index.create_char_indices(train_data, pickle_file_out)

        with open(pickle_file_out, 'rb') as f:
            loaded_char_indices = pickle.load(f, encoding="utf-8")

        print(loaded_char_indices.get_char_index("他"))  # returns 283 for data/train.csv
        print(loaded_char_indices.get_char_bigram_index("シャ"))  # returns 2149 for data/train.csv
        print(loaded_char_indices[0, "unigram"])  # returns 'M' for data/train.csv
        print(loaded_char_indices[0, "bigram"])  # returns '<s>M' for data/train.csv

        return loaded_char_indices

    @staticmethod
    def make_iso_codes():
        iso_codes = ISOCodeToLanguage()
        return iso_codes
