import ignnition
import matplotlib.pyplot as plt
from csv import reader
import os
import numpy as np
from statistics import mean
import time
import tensorflow as tf


def predict(model):
    model = ignnition.create_model(model_dir='./')

    start = time.time()
    predictions = model.predict()
    end = time.time()
    print("Time elapsed: ", end - start)

    predictions = np.array(predictions)
    predictions = predictions[:, :, 0]

    non_binary_values = 0
    for test_sample in predictions:
        print("===========================================================")
        test_sample_rounded = [round(x) for x in test_sample]
        for x in test_sample_rounded:
            if x != 0 and x != 1:
                non_binary_values += 1
        print(test_sample_rounded)

    if non_binary_values != 0:
        print("ERROR: There should be no nonbinary values in the predictions since we are rounding sigmoid to int")

model = ignnition.create_model(model_dir='./')

#predict(model)

