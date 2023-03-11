# CSE573 Language Identifier

## Environment Setup

The default python version for this project environment is `Python 3.6.8`

1. Create a fresh conda environment using `environment.yml`

```
conda env create -f environment.yml
```

2. To activate and deactivate the conda environment for this project:

```
conda activate langid
conda deactivate
```

3. If the `environment.yml` file has changed, update the environment by running the following (after activating the environment):

```
conda env update -f environment.yml
```

4. If any additional package is needed, install the package with `conda install PACKAGE_NAME`, and then create a new `environment.yml` file:

```
conda env export > environment.yml
```



## Pipeline

Note: All scripts must be run from the home directory (topmost directory)

1. Split the data into training and testing sets

```
python3 src/preprocess/train_test_split.py
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

