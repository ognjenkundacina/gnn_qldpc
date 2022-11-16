import numpy as np
import time
from global_variables import *
from read_files_helper import read_files
import math


def degeneration_condition_satisfied(error_bit_values, output_sigm_mu_Vector, H_orth):
    error_bit_values = np.array(error_bit_values)
    output_sigm_mu_Vector = np.array(output_sigm_mu_Vector)
    e = error_bit_values + output_sigm_mu_Vector

    e = np.mod(e, 2)

    #e = tf.transpose(e)  # [[0] [0] [0] ... [1] [1] [1]]

    #e = tf.cast(e, tf.int16)

    file1 = open(str(data_dir) + "/m_n_lv.txt", 'r')
    lines = file1.readlines()
    n = int(lines[2])

    zeroMatrix = np.zeros([n, n])
    identityMatrix = np.identity(n)
    M = np.bmat([[zeroMatrix, identityMatrix], [identityMatrix, zeroMatrix]])

    #print(M.shape) #(116, 116)

    HM = np.matmul(H_orth, M)
    HM = np.round(HM) # za svaki slucaj, da se ne desi da je 0.9999 kastovano u 0
    # print(HM.shape) #(74, 116)
    #HM = tf.convert_to_tensor(HM, dtype=tf.int16)  #[[0 0 0 0 0 0 0 0...

    result = np.matmul(HM, e) # [[2] [4] [1] [1] [1]...

    result = np.mod(result, 2) # [[0] [0] [1] [1] [1]...

    result = np.sum(result) # neki skalar, npr 45

    #tf.print(result, summarize=-1)

    #result = tf.math.floormod(result, 2)

    print(result)
    print(abs(result) <= 0.01)
    print("================")

    return abs(result) <= 0.01


def predict(model):
    start = time.time()
    predictions = model.predict(num_predictions  = 10)
    end = time.time()
    print("Time elapsed: ", end - start)

    predictions = np.array(predictions)
    #predictions = predictions[:, :, 0]


    error_vectors, syndrom_vectors, H, H_orth, m1, m2, n, lv = read_files(data_dir)

    error_vectors_test_labels = error_vectors[split_index:split_index+10]
    print(len(error_vectors_test_labels))
    print(len(predictions))

    if len(predictions) != len(error_vectors_test_labels):
        print("ERROR in predict_helper.py: len(predictions) != len(error_vectors_test_labels)")
        return

    non_binary_values = 0
    areSameCount = 0
    examples_degeneration_condition_satisfied = 0
    for iSample, test_sample in enumerate(predictions):
        test_sample_rounded = [int(round(x)) for x in test_sample]
        for x in test_sample_rounded:
            if x != 0 and x != 1:
                non_binary_values += 1

        error_vector = error_vectors_test_labels[iSample]
        error_vector = [int(round(x)) for x in error_vector]
        areSame = True
        for testBit, errorBit in zip(test_sample_rounded,error_vector):
            if testBit != errorBit:
                areSame = False
                break
        if areSame:
            areSameCount += 1

        if degeneration_condition_satisfied(error_vector, test_sample_rounded, H_orth):
            examples_degeneration_condition_satisfied += 1

    print("The error probability over the test set is : ", 1.0 - (1.0 * areSameCount) / len(predictions))
    print("The degeneracy error probability over the test set is : ", 1.0 - (1.0 * examples_degeneration_condition_satisfied) / len(predictions))
    if non_binary_values != 0:
        print("ERROR: There should be no nonbinary values in the predictions since we are rounding sigmoid to int")
