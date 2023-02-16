import csv
import sys
import pandas

import pandas as pd


class CharIndex:


    def __init__(self, data_file_path):

        self.data = None

        file = open(data_file_path)
        self.data = csv.reader(file)

        self.traverse_data()

        file.close()




    def traverse_data(self):

        for row in self.data:
            text = row[-1]
            print(text)






if __name__ == '__main__':
    data_file_path = sys.argv[1]

    char_indices = CharIndex(data_file_path)



