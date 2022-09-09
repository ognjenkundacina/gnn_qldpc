import numpy as np
import time
from global_variables import *
from read_files_helper import read_files

def predict(model):
    start = time.time()
    predictions = model.predict()
    end = time.time()
    print("Time elapsed: ", end - start)

    predictions = np.array(predictions)
    #predictions = predictions[:, :, 0]


    error_vectors, syndrom_vectors, H, H_orth, m1, m2, n, lv = read_files(data_dir)

    error_vectors_test_labels = error_vectors[split_index:]

    if len(predictions) != len(error_vectors_test_labels):
        print("ERROR in predict_helper.py: len(predictions) != len(error_vectors_test_labels)")
        return

    non_binary_values = 0
    areSameCount = 0
    for iSample, test_sample in enumerate(predictions):
        test_sample_rounded = [round(x) for x in test_sample]
        for x in test_sample_rounded:
            if x != 0 and x != 1:
                non_binary_values += 1

        error_vector = error_vectors_test_labels[iSample]
        areSame = True
        for testBit, errorBit in zip(test_sample_rounded,error_vector):
            if testBit != errorBit:
                areSame = False
                break
        if areSame:
            areSameCount += 1

    print("The error probability over the test set is : ", 1.0 - (1.0 * areSameCount) / len(predictions))
    if non_binary_values != 0:
        print("ERROR: There should be no nonbinary values in the predictions since we are rounding sigmoid to int")
