import pickle
import os
from src.preprocess import char_index


class TopKCharsAndTerms:

    def __init__(self, indices_pickle_file, k=30):
        self.indices = self.open_indices(indices_pickle_file)

        self.char_counts, self.char_bigram_counts, self.term_counts = self.indices.get_counts()

        self.topk_chars = set()
        self.bottomk_chars = set()

        self.topk_bigrams = set()
        self.bottomk_bigrams = set()

        self.topk_terms = set()
        self.bottomk_terms = set()

        self._create_top_k_bottom_k(k)


    def _create_top_k_bottom_k(self, k=30):
        sorted_chars_counts = sorted(self.char_counts.items(), key=lambda item: item[1], reverse=True)
        sorted_bigram_counts = sorted(self.char_bigram_counts.items(), key=lambda item: item[1], reverse=True)
        sorted_term_counts = sorted(self.term_counts.items(), key=lambda item: item[1], reverse=True)

        topk_chars = [item[0] for item in sorted_chars_counts[:k]]
        bottomk_chars = [item[0] for item in sorted_chars_counts[-k:]]

        topk_bigrams = [item[0] for item in sorted_bigram_counts[:k]]
        bottomk_bigrams = [item[0] for item in sorted_bigram_counts[-k:]]

        topk_terms = [item[0] for item in sorted_term_counts[:k]]
        bottomk_terms = [item[0] for item in sorted_term_counts[-k:]]

        self.topk_chars = set(self._add_ties(topk_chars, sorted_chars_counts, k, 'top'))
        self.bottomk_chars = set(self._add_ties(bottomk_chars, sorted_chars_counts, k, 'bottom'))

        self.topk_bigrams = set(self._add_ties(topk_bigrams, sorted_bigram_counts, k, 'top'))
        self.bottomk_chars = set(self._add_ties(bottomk_bigrams, sorted_bigram_counts, k, 'bottom'))

        self.topk_terms = set(self._add_ties(topk_terms, sorted_term_counts, k, 'top'))
        self.bottomk_terms = set(self._add_ties(bottomk_terms, sorted_term_counts, k, 'bottom'))


    def _add_ties(self, cur_k_set, sorted_counts, k, k_type):
        index = k + 1
        if k_type == "bottom":
            cur_lowest_k = sorted_counts[-k][1]
            lowest_k_value = cur_lowest_k

            while (index <= len(sorted_counts)) and (lowest_k_value == cur_lowest_k):
                cur_str = sorted_counts[-index][0]
                cur_k_set.append(cur_str)
                index += 1
                cur_lowest_k = sorted_counts[-index][1]

            return cur_k_set

        elif k_type == "top":
            cur_highest_k = sorted_counts[k - 1][1]
            highest_k_value = cur_highest_k

            while (index < len(sorted_counts) - 1) and (highest_k_value == cur_highest_k):
                cur_str = sorted_counts[index][0]
                cur_k_set.append(cur_str)
                index += 1
                cur_highest_k = sorted_counts[index][1]

            return cur_k_set

        raise ValueError(f"type argument should be 'top' or 'bottom', but got {type}")



    def save(self, output_pickle_file):
        with open(output_pickle_file, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    def term_in_bottomk(self, term):
        """
        :param term: Either the term as a string or its corresponding index
        :return: A boolean, true if the given term is in the bottomk
        """
        return term in self.bottomk_terms


    def term_in_topk(self, term):
        return term in self.topk_terms


    def char_in_bottomk(self, char):
        return char in self.bottomk_chars

    def char_in_topk(self, char):
        return char in self.topk_chars

    def char_bigram_in_bottomk(self, bigram):
        return bigram in self.bottomk_bigrams


    def char_bigram_in_topk(self, bigram):
        return bigram in self.topk_bigrams

    @staticmethod
    def open_indices(pickle_file_out):
        with open(pickle_file_out, 'rb') as f:
            loaded_indices = pickle.load(f, encoding="utf-8")
        return loaded_indices


    @staticmethod
    def create_topk_chars_and_terms(topk_pickle_file, indices_pickle_file):
        if not os.path.isfile(topk_pickle_file):
            topk_chars_terms = TopKCharsAndTerms(indices_pickle_file)
            topk_chars_terms.save(topk_pickle_file)

        with open(topk_pickle_file, 'rb') as f:
            loaded_topk = pickle.load(f, encoding="utf-8")

        return loaded_topk
