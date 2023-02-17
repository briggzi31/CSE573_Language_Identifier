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

