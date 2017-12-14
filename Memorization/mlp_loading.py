# -*- coding: utf-8 -*-
"""
Spyder Editor

Sacha Perry-Fagant
"""

import tensorflow.examples.tutorials.mnist.input_data as input_data

def get_mnist():
    m=input_data.read_data_sets("MNIST")
    imgs = m.train.images.reshape(-1, 28, 28)
    labels = m.train.labels
    test = m.test.images.reshape(-1, 28, 28)
    t_labels = m.test.labels
    valid = m.validation.images.reshape(-1,28,28)
    v_labels = m.validation.labels
    return imgs, labels, valid, v_labels, test, t_labels

# Changes the images to be a long array instead of a matrix
def flatten(data):
    all_data = []
    for d in data:
        dat = d.reshape(1, len(d) * len(d[0]))
        all_data.append(dat[0].tolist())
    return all_data
