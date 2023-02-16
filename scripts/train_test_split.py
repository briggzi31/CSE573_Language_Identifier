import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_data = '/data/train.csv'
test_data = '/data/test.csv'

data = pd.read_csv("../data/sentences.csv")
train, test = train_test_split(data, test_size=0.2, random_state=1)

train.to_csv(train_data)
test.to_csv(test_data)