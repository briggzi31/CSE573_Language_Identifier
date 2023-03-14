import pandas as pd
import sys
from sklearn.model_selection import train_test_split

sentences = sys.argv[1]
train_data = sys.argv[2]
test_data = sys.argv[3]

data = pd.read_csv(sentences)
train, test = train_test_split(data, test_size=0.2, random_state=1)

train.to_csv(train_data)
test.to_csv(test_data)


