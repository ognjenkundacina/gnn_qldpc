import sys

import tensorflow as tf
import os
from csv import reader
import numpy as np

import math

from global_variables import *


#@tf.function()
def CustomLoss(y_true, y_pred):

    error_bit_values = tf.expand_dims(y_pred[:, -1], axis=0)
    output_sigm_mu_Vector = tf.expand_dims(y_pred[:, -2], axis=0)
    e = error_bit_values + output_sigm_mu_Vector
    e = tf.transpose(e)
    e = tf.abs(tf.sin(e * math.pi / 8.0))
    #e=tf.math.floormod(e,2)

    #print(error_bit_values.shape)
    #print(output_sigm_mu_Vector.shape)
    #print(e.shape)
    #print(tf.get_static_value(error_bit_values))
    #print(tf.get_static_value(output_sigm_mu_Vector))
    #print(tf.get_static_value(e))


    H_orth = np.genfromtxt(str(data_dir) + '/H_orth_Random_QLDPC.csv', delimiter=',')
    file1 = open(str(data_dir) + "/m_n_lv.txt", 'r')
    lines = file1.readlines()
    n = int(lines[2])

    zeroMatrix = np.zeros([n, n])
    identityMatrix = np.identity(n)
    M = np.bmat([[zeroMatrix, identityMatrix], [identityMatrix, zeroMatrix]])

    HM = np.matmul(H_orth, M)
    HM = np.round(HM)  # za svaki slucaj, da se ne desi da je 0.9999 kastovano u 0
    HM = tf.convert_to_tensor(HM, dtype=tf.float32)

    result = tf.matmul(HM, e)
    result = tf.abs(tf.sin(result * math.pi / 8.0))
    #tf.print(result, summarize=-1)

    result = tf.reduce_sum(result)
    #result = tf.reduce_sum(result) - tf.reduce_sum(tf.pow(output_sigm_mu_Vector - 0.5, 2))

    return result


#@tf.function()
def sum_of_bits(y_true, y_pred):
    error_bit_values = tf.expand_dims(y_pred[:, -1], axis=0) #[[0 0 0 ... 1 0 0]]
    output_sigm_mu_Vector = tf.expand_dims(y_pred[:, -2], axis=0) # [[0.493709594 0.432534277 0.445749462 ... 0.11966759 0.118129909 0.175256938]]

    error_bit_values = tf.cast(tf.round(error_bit_values), tf.int16) # [[0 0 0 ... 1 0 0]]
    output_sigm_mu_Vector = tf.cast(tf.round(output_sigm_mu_Vector), tf.int16) # [[0 0 0 ... 0 0 1]]

    e = error_bit_values + output_sigm_mu_Vector
    e = tf.math.floormod(e, 2)
    e = tf.transpose(e)  # [[0] [0] [0] ... [1] [1] [1]]

    H_orth = np.genfromtxt(str(data_dir) + '/H_orth_Random_QLDPC.csv', delimiter=',')
    #print(H_orth.shape) #(74, 116)

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
    HM = tf.convert_to_tensor(HM, dtype=tf.int16)  #[[0 0 0 0 0 0 0 0...

    result = tf.matmul(HM, e) # [[2] [4] [1] [1] [1]...

    result = tf.math.floormod(result, 2) # [[0] [0] [1] [1] [1]...

    result = tf.reduce_sum(result) # neki skalar, npr 45

    #tf.print(result, summarize=-1)

    #result = tf.math.floormod(result, 2)

    return result