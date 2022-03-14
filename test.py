import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.models import load_model

Channels=21
batch_size=21
time_steps = 1
features = 105


def test_model(test_data, model = False):
    X=np.array([i[0] for i in test_data]).reshape([-1,time_steps,features])
    # X=np.array(input_data).reshape([-1,21,2500,1])
    X = np.asarray(X).astype('float64')
    Y=[i[1] for i in test_data]
    Y=np.array(Y)
    print(X.shape,Y.shape)
    if not model:
        model = load_model('model_LSTM_1')

    return model.evaluate(X,Y, verbose=2)

testing_data=np.load('data/test/testing_data.npy',allow_pickle=True)

score = test_model(testing_data)
print('The accuracy is: ', score)