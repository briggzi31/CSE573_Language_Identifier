"""
    This module reads in a .csv file from a specified path and saves a CharIndex object to the specified path
    To use the CharIndex object in another model:

    #############################

    import pickle
    from src.preprocess.char_index import CharIndex

    with open([path_to_pickle_file], 'rb') as f:
        loaded_char_indices = pickle.load(f, encoding="utf-8")

    print(loaded_char_indices.get_char_index("他"))  # returns 283 for data/train.csv
    print(loaded_char_indices.get_char_bigram_index("シャ"))  # returns 2149 for data/train.csv
    print(loaded_char_indices[0, "unigram"])  # returns 'M' for data/train.csv
    print(loaded_char_indices[0, "bigram"])  # returns '<s>M' for data/train.csv

    #############################
"""

import csv
import pickle
import re

class CharIndex:


    def __init__(self, data_file_path):
        self.char_to_index = {}
        self.index_to_char = []

        self.charbigram_to_index = {}
        self.index_to_charbigram = []

        self.term_to_index = {}
        self.index_to_term = []

        self._char_counter = {}
        self._char_bigram_counter = {}
        self._term_counter = {}

        self.traverse_data(data_file_path)


    def traverse_data(self, data_file_path):
        file = open(data_file_path)
        data = csv.reader(file)

        header = True
        for row in data:
            if header:
                header = False
                continue

            text = row[-1].strip()

            self._char_level_indices(text)
            self._term_level_indices(text)

        file.close()


    def _term_level_indices(self, text):
        text = text.split()

        for term in text:
            if term not in self.term_to_index:
                self.term_to_index[term] = len(self.index_to_term)
                self.index_to_term.append(term)

            # counting terms
            if term not in self._term_counter:
                self._term_counter[term] = 0
            self._term_counter[term] += 1

    def _char_level_indices(self, text):
        """
        :param text: reads in the text to create indices of char unigrams and bigrams
        :return: None
        """
        # special case, BOS token
        prev_char = "<s>"
        for char in text:

            # add current char and its corresponding index
            if char not in self.char_to_index:
                self.char_to_index[char] = len(self.index_to_char)
                self.index_to_char.append(char)

            # add current bigram and its corresponding index
            cur_bigram = prev_char + char
            if cur_bigram not in self.charbigram_to_index:
                self.charbigram_to_index[cur_bigram] = len(self.index_to_charbigram)
                self.index_to_charbigram.append(cur_bigram)

            # counting bigrams
            if cur_bigram not in self._char_bigram_counter:
                self._char_bigram_counter[cur_bigram] = 0
            self._char_bigram_counter[cur_bigram] += 1

            # counting chars without whitespace
            if re.search(r"\s", char) is None:
                if char not in self._char_counter:
                    self._char_counter[char] = 0
                self._char_counter[char] += 1

            prev_char = char

        # special case, EOS token
        cur_bigram = prev_char + '</s>'
        if cur_bigram not in self.charbigram_to_index:
            self.charbigram_to_index[cur_bigram] = len(self.index_to_charbigram)
            self.index_to_charbigram.append(cur_bigram)

        # counting bigrams
        if cur_bigram not in self._char_bigram_counter:
            self._char_bigram_counter[cur_bigram] = 0
        self._char_bigram_counter[cur_bigram] += 1


    def get_char_index(self, char):
        """
        :param char: a string of length 1
        :return: the int corresponding to the given char
            if char not seen before, returns -1
        """
        # check if char
        if not (isinstance(char, str) and len(char) == 1):
            raise ValueError(f"char argument not type char, instead {char}")

        try:
            return self.char_to_index[char]
        except:
            return -1


    def get_char_bigram_index(self, bigram):
        """
        :param char: a string of length 1
        :return: the int corresponding to the given char bigram
            if char bigram not seen before, returns -1
        """
        # check if bigram
        if not (isinstance(bigram, str) and len(bigram) == 2) and \
                ("<s>" not in bigram) and ("</s>" not in bigram):
            raise ValueError(f"char argument not type char, instead {bigram}")

        try:
            return self.charbigram_to_index[bigram]
        except:
            return -1

    def get_term_index(self, term):
        """
        :param term: a term as a string
        :return: the int corresponding to the given term
            if term not seen before, returns -1
        """
        #check if term
        if not (isinstance(term, str)):
            raise ValueError(f"term argument not type str, instead got {term}")

        try:
            return self.term_to_index[term]
        except:
            return -1


    def __getitem__(self, item):
        """
        :param item: int index, and the type of character/term gram as a string, i.e. "unigram", "bigram", "term"
        :return:
            The corresponding character/term gram mapped to the given integer index
            If given index not associated with a character/term, returns -1
        """
        if len(item) != 2:
            raise ValueError(
                f"argument must contain int index and either a "
                f"unigram char or bigram char as a string, but got {item}"
            )

        index = item[0]
        gram = item[1]

        if gram == "bigram":
            try:
                return self.index_to_charbigram[index]
            except:
                return -1
        elif gram == "unigram":
            try:
                return self.index_to_char[index]
            except:
                return -1
        elif gram == "term":
            try:
                return self.index_to_term[index]
            except:
                return -1

        # something is wrong
        raise ValueError(
                f"Unable to return the char or term. Argument should be index and either"
                f"'unigram', 'bigram', or 'term', but received the argument {item}"
        )


    def get_counts(self):
        return self._char_counter, self._char_bigram_counter, self._term_counter

    def save(self, output_file_path):
        """
        :param output_file_path: The file path where the pickle file should be stored
            Saves the instance to a pickle file to be loaded in later :)
        :return: None
        """
        with open(output_file_path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)


def create_char_indices(data_file_path, pickle_file_path):
    char_indices = CharIndex(data_file_path)
    char_indices.save(pickle_file_path)
