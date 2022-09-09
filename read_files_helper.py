from csv import *
import numpy as np
from global_variables import *

def read_files(data_dir):

    print("reading files started")

    path = str(data_dir) + "/Train_error_full_trans.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        error_vectors = []
        for i, line in enumerate(csv_reader):
            error_vector = [abs(float(element)) for element in line]
            error_vectors.append(error_vector)

    path = str(data_dir) + "/Train_syndrome_full_trans.csv"
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        syndrom_vectors = []
        for i, line in enumerate(csv_reader):
            syndrom_vector = [abs(float(element)) for element in line]
            syndrom_vectors.append(syndrom_vector)

    H = np.genfromtxt(str(data_dir) + '/Random_QLDPC_H.csv', delimiter=',')
    H_orth = np.genfromtxt(str(data_dir) + '/H_orth_Random_QLDPC.csv', delimiter=',')

    file1 = open(str(data_dir) + "/m_n_lv.txt", 'r')
    lines = file1.readlines()
    m1 = int(lines[0])
    m2 = int(lines[1])
    n = int(lines[2])
    lv = float(lines[3])

    #primjer za najmanji kod
    # m = 21
    # n = 58
    #print(H.shape) #(42, 116)
    #print(H_orth.shape) # (74, 116)
    #print(len(error_vectors[0])) # 116
    #print(len(syndrom_vectors[0])) # 42

    if H.shape[0] != (m1 + m2):
        print("ERROR: H.shape[0] != 2 * (m1 + m2) ")
        return

    if H.shape[1] != 2 * n:
        print("ERROR: H.shape[1] != 2 * n")
        return

    if H.shape[0] != len(syndrom_vectors[0]):
        print("ERROR: H.shape[0] != len(syndrom_vectors[0])")
        return

    if H.shape[1] != len(error_vectors[0]):
        print("ERROR: H.shape[1] != len(error_vectors[0])")
        return

    if H.shape[1] != H_orth.shape[1]:
        print("ERROR: H.shape[1] != H_orth.shape[1]")
        return

    if H.shape[0] + H_orth.shape[0] != H.shape[1]:
        print("ERROR: H.shape[0] + H_orth.shape[0] != H.shape[1]")
        return

    if len(error_vectors) != len(syndrom_vectors):
        print("ERROR: len(error_vectors) != len(syndrom_vectors)")
        return

    print("reading files done")

    return error_vectors, syndrom_vectors, H, H_orth, m1, m2, n, lv


def transpose_error_and_syndrom_files():
    import pandas as pd

    print("Transposing input files started")

    pd.read_csv(str(data_dir) + "/Train_error_full.csv", header=None).T.to_csv(str(data_dir) + "/Train_error_full_trans.csv", header=False, index=False)
    pd.read_csv(str(data_dir) + "/Train_syndrome_full.csv", header=None).T.to_csv(str(data_dir) + "/Train_syndrome_full_trans.csv", header=False, index=False)

    print("Transposing input files finished")