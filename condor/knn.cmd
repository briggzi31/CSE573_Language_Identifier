executable = /home2/lexwang/cse573/knn.sh
getenv = true
output = acc_knn
error = knn.err
log = knn.log
notification = complete
arguments = "/home2/lexwang/pickle_objects/features/train_vectors.pickle /home2/lexwang/pickle_objects/gold_labels/train_gold_labels.pickle /home2/lexwang/pickle_objects/features/dev_vectors.pickle /home2/lexwang/pickle_objects/gold_labels/dev_gold_labels.pickle /home2/lexwang/pickle_objects/features/test_vectors.pickle /home2/lexwang/pickle_objects/gold_labels/test_gold_labels.pickle /home2/lexwang/cse573/knn_out.txt"
transfer_executable = false
request_memory = 4096
request_GPUs = 1
queue
