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


class CharIndex:


    def __init__(self, data_file_path):
        self.char_to_index = {}
        self.index_to_char = []

        self.charbigram_to_index = {}
        self.index_to_charbigram = []

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

                prev_char = char

            # special case, EOS token
            cur_bigram = prev_char + '</s>'
            if cur_bigram not in self.charbigram_to_index:
                self.charbigram_to_index[cur_bigram] = len(self.index_to_charbigram)
                self.index_to_charbigram.append(cur_bigram)

        file.close()

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
        # check if char
        if not (isinstance(bigram, str) and len(bigram) == 2):
            raise ValueError(f"char argument not type char, instead {bigram}")

        try:
            return self.charbigram_to_index[bigram]
        except:
            return -1


    def __getitem__(self, item):
        """
        :param item: int index, and the type of character gram as a string, i.e. "unigram" or "bigram"
        :return:
            The corresponding character gram mapped to the given integer index
            If given index not associated with a character, returns -1
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

        # something is wrong
        raise ValueError(f"Unable to return the char. Received the argument {item}")


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
