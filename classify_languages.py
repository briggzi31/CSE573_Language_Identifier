import sys
# from src.classifiers.naive_bayes import NaiveBayes
from src.classifiers.decision_tree import DecisionTree
from src.baseline.majority_vote import MajorityVote


def make_classifier(
        train_vectors,
        train_gold_labels,
        dev_vectors,
        dev_gold_labels,
        test_vectors,
        test_gold_labels,
        classifier
):
    # check if valid model
    if (classifier != 0) and (classifier != 1) and (classifier != 2) and (classifier != 3):
        raise ValueError(f'classifier argument should be 1, 2, or 3, instead got {classifier}')

    model = None

    if classifier == 0:
        model = MajorityVote(train_vectors, train_gold_labels, dev_vectors,
                             dev_gold_labels, test_vectors, test_gold_labels)
    elif classifier == 1:
        model = DecisionTree(train_vectors, train_gold_labels, dev_vectors,
                             dev_gold_labels, test_vectors, test_gold_labels)
    elif classifier == 2:
        rand = 0
        # model = multinomialNB



    return model


def train_predict(model, use_dev, output):
    model.train()
    model.predict(use_dev, output)



if __name__ == '__main__':
    # pickle files
    # train pickle
    train_vectors_file = "pickle_objects/features/train_vectors.pickle"
    train_gold_labels_file = "pickle_objects/gold_labels/train_gold_labels.pickle"

    # dev pickle
    dev_vectors_file = "pickle_objects/features/dev_vectors.pickle"
    dev_gold_labels_file = "pickle_objects/gold_labels/dev_gold_labels.pickle"

    # test pickle
    test_vectors_file = "pickle_objects/features/test_vectors.pickle"
    test_gold_labels_file = "pickle_objects/gold_labels/test_gold_labels.pickle"

    # train data k1 = k_char, k2 = k_word, k3 = k_char_bg hyperparameters
    k1 = sys.argv[1]
    k2 = sys.argv[2]
    k3 = sys.argv[3]

    # model hyper parameters
    classifier = int(sys.argv[4])
    use_dev = bool(sys.argv[5])

    # output file for accuracy
    output_accuracy_file = sys.argv[6]

    params = "_k1=" + str(k1) + "_k2=" + str(k2) + "_k3=" + str(k3)

    train_vectors_file += params
    train_gold_labels_file += params
    dev_vectors_file += params
    dev_gold_labels_file += params
    test_vectors_file += params
    test_gold_labels_file += params

    # create model, fit model, and predict
    model = make_classifier(train_vectors_file, train_gold_labels_file, dev_vectors_file,
                            dev_gold_labels_file, test_vectors_file, test_gold_labels_file,
                            classifier
                            )
    train_predict(model, use_dev, output_accuracy_file)
