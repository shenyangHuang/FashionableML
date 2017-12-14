# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:40:39 2017

@author: Sacha Perry-Fagant
"""
import random
import copy

# Takes in the labels
# Can select with which probability labels are changed
# 25% and 50% were mentioned in the paper
def shuffle_labels(labelled_data, probability = 0.25, labels = None):
    # If the labels are not provided
    if labels == None:
        labels = list(set(labelled_data))
    # Make sure not to overwrite the original labels
    y = copy.copy(labelled_data)
    for i in range(len(y)):
        p = float(random.randint(0,100))/100
        if p < probability:
            # randomly choose a new label
            label = random.choice(labels)      
            # make sure the new label is different from the old one
            while y[i] == label:                
                label = random.choice(labels)                
            y[i] = label
    return y

# every 4 elements is randomized
def randomize(labelled_data, probability = 4, labels = None):
    # If the labels are not provided
    if labels == None:
        labels = list(set(labelled_data))
    # Make sure not to overwrite the original labels
    y = copy.copy(labelled_data)
    for i in range(len(y)):
        if i % 4 == 0:
            # randomly choose a new label
            label = random.choice(labels)      
            # make sure the new label is different from the old one
            while y[i] == label:                
                label = random.choice(labels)                
            y[i] = label
    return y