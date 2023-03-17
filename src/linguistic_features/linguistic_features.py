import pickle
import os
import csv
import numpy as np
from collections import Counter
from src.preprocess import char_index
from src.preprocess.iso_code_to_language import ISOCodeToLanguage
from src.preprocess.topk_chars_terms import TopKCharsAndTerms


class LinguisticFeatures:

    def __init__(self, train_data_path, indices_pickle_file, topk_pickle_file):
        # file paths
        self.train_data_path = train_data_path
        self.indices_pickle_file = indices_pickle_file
        self.topk_pickle_file = topk_pickle_file

        self.train_data = self.process_data(train_data_path)
        self.loaded_char_indices = None
        self.topk_chars_terms = None

        self.gold_labels = []
        self.feature_vectors = self.get_linguistic_features()


    def process_data(self, train_data_path):
        """
        :param train_data_path:  takes in a path to a csv file
        :return: a list of instances, where each instance is a list as follows:
                0           1         2
            instance_id, iso_code, raw_text
        """
        data = []
        with open(train_data_path, 'r') as file:
            csv_reader = csv.reader(file)

            header = True
            id = 1
            for instance in csv_reader:
                # for header
                if header:
                    header = False
                    continue

                new_inst = [id]
                new_inst.extend(instance[-2:])
                data.append(tuple(new_inst))
                id += 1
        return data

    def get_linguistic_features(self, k_char=3, k_word=3, k_char_bg=3):
        """
        :param:
            int k_char (top-k char), int k_word (top-k term), int k_char_bg (top-k char bigrams)

        Iterate through the dataset and compute linguistic feature vectors for each document in data.

        :returns:
            a numpy array of numpy array, where cell a_ij corresponds to feature j in instance i
        """
        # Load char-to-index mapping from pickle file
        # if train data, creates the pickle file
        # if test data, reads from pickle fle
        self.loaded_char_indices = self.create_char_indices(self.train_data_path, self.indices_pickle_file)
        # Load topk_chars_terms sets over all training data from pickle file
        # if train data, creates the pickle file
        # if test data, reads from pickle fle
        self.topk_chars_terms = TopKCharsAndTerms.create_topk_chars_and_terms(
                self.topk_pickle_file, self.indices_pickle_file)

        features = []

        # Process each instance/doc in train data
        for id, iso_code, text in self.train_data:
            word_char_features = []

            self.gold_labels.append(iso_code)

            char_features = self.get_char_level_features(text, k_char, k_char_bg)  # This function doesn't use id and
                            # iso_code, it just returns a list of numbers (feature vector) for this line of text
            word_features = self.get_word_level_features(text, k_word)

            word_char_features.extend(char_features)
            word_char_features.extend(word_features)
            word_char_features = np.array(word_char_features)
            features.append(word_char_features)

        return np.array(features)

    def get_word_level_features(self, text, k=3):
        """
        Computes all linguistic features at word-level.
            Linguistic features computed:
                (1) Top k frequent words (default k is 3)
                (2) Average length of the full word
                (3) Indicator whether given text contains topk term from whole training data
                (4) Indicator whether given text contains bottomk term from whole training data

        :returns: A word_level feature vector as a list:
            [top1term, top2term, ..., topkterm, avgtermlength, contains_topk_term, contains_bottomk_term]
        """
        word_features = []

        text = text.strip().split()
        term_counter = {}
        tot_length = 0

        contains_term_in_topk = False
        contains_term_in_bottomk = False

        for term in text:
            tot_length += len(term)

            if term not in term_counter:
                term_counter[term] = 0
            term_counter[term] += 1


            if self.topk_chars_terms.term_in_topk(term):
                contains_term_in_topk = True
            if self.topk_chars_terms.term_in_bottomk(term):
                contains_term_in_bottomk = True

        # gets top k terms
        term_counter = sorted(term_counter.items(), key=lambda item: item[1])[:k]

        top_k_terms = [self.loaded_char_indices.get_term_index(term[0]) for term in term_counter]

        top_k_terms = LinguisticFeatures.pad_list(top_k_terms, k)

        word_features.extend(top_k_terms)

        # gets avg_length
        avg_length = tot_length / len(text)
        word_features.append(avg_length)

        if contains_term_in_topk:
            word_features.append(1)
        else:
            word_features.append(0)

        if contains_term_in_bottomk:
            word_features.append(1)
        else:
            word_features.append(0)

        return word_features

    def get_char_level_features(self, text, k_char, k_char_bg):
        """
        Computes all linguistic features at character-level.
            Linguistic features computed:
                (1) Top k frequent characters
                (2) Ratio of whitespace to number of chars in doc
                (3) Ratio of alphanumeric chars to all chars
                (4) Ratio of lower-case and upper-case chars to all chars
                (5) Top k frequent bigrams

        :returns: char_features; List; each item is either an int (representing index) or float (ratio)
                                 The returned list should have length = k_char + k_char_bg + 4
        """
        char_features = []

        chars = []  # Ignore whitespace
        char_bgs = []

        char_in_topk = False
        char_in_bottomk = False
        bigram_in_topk = False
        bigram_in_bottomk = False

        # for bigrams
        # for BOS padding
        char_bgs.append("<s>"+text[0])
        for i in range(0, len(text)):
            unigram = text[i]
            if i+1 < len(text):
                bigram = text[i]+text[i+1]
                char_bgs.append(bigram)

            else:  # i == len(text)
                bigram = text[i]+"</s>"
                char_bgs.append(bigram)
            chars.append(unigram)

            if self.topk_chars_terms.char_in_topk(unigram):
                char_in_topk = True
            if self.topk_chars_terms.char_in_bottomk(unigram):
                char_in_topk = True
            if self.topk_chars_terms.char_bigram_in_topk(bigram):
                bigram_in_topk = True
            if self.topk_chars_terms.char_bigram_in_bottomk(bigram):
                bigram_in_bottomk = True

        char_bg_indices = [self.loaded_char_indices.get_char_bigram_index(cbg) for cbg in char_bgs]
        char_indices = [self.loaded_char_indices.get_char_index(c) for c in chars]

        whitespace_idx = self.loaded_char_indices.get_char_index(' ')  # Index of whitespace

        # Find top k frequent characters
        top_k_char_indices = self.get_top_k_chars(char_indices, whitespace_idx, k_char)
        char_features.extend(top_k_char_indices)

        top_k_char_bigram_counter = Counter(char_bg_indices).most_common(k_char_bg)
        top_k_char_bigram_indices = [idx for idx, count in top_k_char_bigram_counter]
        top_k_char_bigram_indices = LinguisticFeatures.pad_list(top_k_char_bigram_indices, k_char_bg)
        char_features.extend(top_k_char_bigram_indices)

        # Calculate ratio of whitespace to number of chars
        chars_count = Counter(char_indices)
        whitespace_ratio = chars_count[whitespace_idx] / sum(chars_count.values())
        char_features.append(whitespace_ratio)

        alnum_count = 0
        upper_count = 0
        lower_count = 0
        for char_idx in chars_count.keys():
            if self.loaded_char_indices[char_idx, "unigram"].isalnum():
                alnum_count += chars_count[char_idx]
            if self.loaded_char_indices[char_idx, "unigram"].isupper():
                upper_count += chars_count[char_idx]
            if self.loaded_char_indices[char_idx, "unigram"].islower():
                lower_count += chars_count[char_idx]

        # Calculate ratio of alphanumeric chars to all chars
        alnum_ratio = alnum_count / sum(chars_count.values())
        char_features.append(alnum_ratio)

        # Calculate ratio of upper-case to all chars
        upper_ratio = upper_count / sum(chars_count.values())
        char_features.append(upper_ratio)

        # Calculate ratio of lower-case to all chars
        lower_ratio = lower_count / sum(chars_count.values())
        char_features.append(lower_ratio)

        if char_in_topk:
            char_features.append(1)
        else:
            char_features.append(0)
        if char_in_bottomk:
            char_features.append(1)
        else:
            char_features.append(0)
        if bigram_in_topk:
            char_features.append(1)
        else:
            char_features.append(0)
        if bigram_in_bottomk:
            char_features.append(1)
        else:
            char_features.append(0)

        return char_features

    def save(self, output_pickle_file):
        with open(output_pickle_file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def pad_list(lst, k):
        if len(lst) < k:
            for i in range(k - len(lst)):
                lst.append(-1)
        return lst

    @staticmethod
    def get_top_k_chars(char_indices, whitespace_idx, k):
        """
        Find the n most frequent characters in a document (excluding whitespace), and return the indices of the n most
        frequent characters in a list.

        :param doc: list; each doc is one row in the train or test data, where the item at index 2 is ISO code of the
                    language, and the item at index 3 is the text of the document
        :return: list; in which each item is the index of a most frequent char in doc
        """
        char_indices_without_space = [i for i in char_indices if i is not whitespace_idx]

        top_n_chars = Counter(char_indices_without_space).most_common(k)

        top_n_char_indices = [idx for idx, count in top_n_chars]

        # adds -1 if len of text shorter than k
        top_n_char_indices = LinguisticFeatures.pad_list(top_n_char_indices, k)

        return top_n_char_indices

    @staticmethod
    def create_char_indices(train_data_path, indices_pickle_file):
        """
            # print(loaded_char_indices.get_char_index("他"))  # returns 283 for data/train.csv
            # print(loaded_char_indices.get_char_bigram_index("シャ"))  # returns 2149 for data/train.csv
            # print(loaded_char_indices.get_term_index("Kros")) # returns 330 for data/train.csv
            # print(loaded_char_indices[0, "unigram"])  # returns 'M' for data/train.csv
            # print(loaded_char_indices[0, "bigram"])  # returns '<s>M' for data/train.csv
            # print(loaded_char_indices[0, "term"]) # returns 'Mi' for data/train.csv
        :param train_data_path: the file path for the training data
        :param indices_pickle_file: the file path for the indices pickle file
        :return: A src.preprocess.CharIndex object
        """
        if not os.path.isfile(indices_pickle_file):
            char_index.create_char_indices(train_data_path, indices_pickle_file)

        with open(indices_pickle_file, 'rb') as f:
            loaded_char_indices = pickle.load(f, encoding="utf-8")

        return loaded_char_indices

    @staticmethod
    def make_iso_codes():
        iso_codes = ISOCodeToLanguage()
        return iso_codes


# lf = LinguisticFeatures('../../data/train.csv', '../../data/char2idx.pickle')
# lf.get_linguistic_features(3, 3)  # Note: argument k (top k char), argument n (top n char bigrams)


if __name__ == '__main__':
    pass