# CSE573 Language Identifier

## Environment Setup

The default python version for this project environment is `Python 3.6.8`

If you don't have conda installed, the conda install guide is [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Create a fresh conda environment using `environment.yml`

```
conda env create -f environment.yml
```

2. To activate and deactivate the conda environment for this project:

```
conda activate langid
conda deactivate
```

## Dataset

Download the Big Language Detection Dataset [here](https://www.kaggle.com/datasets/chazzer/big-language-detection-dataset?select=sentences.csv).

## Pipeline

Note: All scripts must be run from the home directory (topmost directory)

1. Split the data into training and testing sets

```
./src/scripts/train_test_split.sh data/sentences.csv data/train.csv data/test.csv
```

2. Process training data, which includes the following steps: (1) assign each character with a unique index and save it in a pickle file, (2) find the top *k* characters and terms, save in a pickle file, (3) create feature vectors for each data instance in training set, and split the training set into train and dev sets 

```
./src/scripts/process_data.sh data/train.csv data/test.csv pickle_objects/char_indices.pickle pickle_objects/topk.pickle pickle_objects/features/train_vectors.pickle pickle_objects/gold_labels/train_gold_labels.pickle pickle_objects/features/dev_vectors.pickle pickle_objects/gold_labels/dev_gold_labels.pickle
```

After running the script above, the directory `pickle_objects` should contain the following sub-directories and files:

```
./pickle_objects:
char_indices.pickle 						topk.pickle
features            gold_labels         

./pickle_objects/features:
dev_vectors.pickle   train_vectors.pickle

./pickle_objects/gold_labels:
dev_gold_labels.pickle   train_gold_labels.pickle
```

3. Train the selected model on training data, and predict the languages of the test data. All outputs contain the classification reports will be in the accuracy/ directory. To train and predict:

3.1 To run the baseline model:

```
 ./src/scripts/run_baseline.sh
```

3.2 To run the Multinomial Naive Bayes Model:

```
./src/scripts/run_nb.sh
```

3.3 To run the Decision Tree Model:

```
./src/scripts/run_decision_tree.sh
```

3.4 To run the Logistic Regression Model:

```
.src/scripts/run_logistic_regression.sh
```
