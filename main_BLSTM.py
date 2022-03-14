import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from models import BLSTM as blstm_model

Channels=21
batch_size=21
time_steps = 1
features = 105

def train_model_BLSTM(train_data, model = False):
    X=np.array([i[0] for i in train_data]).reshape([-1,time_steps,features])
    # X=np.array(input_data).reshape([-1,21,2500,1])
    # X = np.asarray(X)
    # X=train_data[0]
    # X=np.array([[X]])
    X = np.asarray(X).astype('float64')
    Y=[i[1] for i in train_data]
    # Y=np.array([train_data[1]])
    
    Y=np.array(Y)
    print(X.shape,Y.shape)
    if not model:
        model = blstm_model.neural_network_BLSTM(time_steps,features)
    
    model.fit(X, Y, epochs=1000)
    return model

training_data=np.load('data/train/training_data_change.npy',allow_pickle=True)  
# shuffle(training_data)
# x=[]
# for i in training_data:
#     print(i[0],i[1])
#     x.append(i[0])
#     x.append(i[1])
#     break
# print(x)
# training_data=np.random.random(size=(1,105))
# ct=0
# for i in training_data:
#     print(i)
#     ct=ct+1
#     if ct==5:
#         break

model = train_model_BLSTM(training_data)
# model.save('model_BLSTM_1')