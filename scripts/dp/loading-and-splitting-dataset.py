#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier




DATASET_DIRECTORY = r'C:\Users\daniela\OneDrive\Documents\geral\pibic\digits'
LABELS = []
IMG_SIZE = 28

DATADIR = r'C:\Users\daniela\OneDrive\Documents\geral\pibic\digits'
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]




# ARRAY DE LABELS BASEADO NO NOME DAS PASTAS
for dir_name in range(len(os.listdir(DATASET_DIRECTORY))):
    LABELS.append(os.listdir(DATASET_DIRECTORY)[dir_name])

print(LABELS)




training_data = []

def create_training_data():
    for category in LABELS:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = LABELS.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                print("erro")
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))




# SEPARANDO IMG E LABEL

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

# print(x[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)




# SALVAR DATASET

pickle_out = open('x.pickle', 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()




# CARREGAR DATASET

pickle_in = open('x.pickle', 'rb')
x = pickle.load(pickle_in)

