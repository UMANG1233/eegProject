from matplotlib.pyplot import axis
import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle

during_data=np.load('./data/during_file_path/entropy_during.npy',allow_pickle=True)
before_data=np.load('./data/before_file_path/entropy_before.npy',allow_pickle=True)

def combine(x):
    res = x.flatten()
    return res

temp=[]
for i in during_data:
    loc=[]
    for j in i:
        loc.append(combine(j))
    temp.append(loc)
during_data=temp
during_data=np.array(during_data)
temp.clear()

for i in before_data:
    loc=[]
    for j in i:
        loc.append(combine(j))
    temp.append(loc)
before_data=temp
before_data=np.array(before_data)


during_result=[]
for i in range(0,during_data.shape[0]):
    temp=[]
    for j in range(0,during_data.shape[1]):
        temp.append([0,1])
    during_result.append(temp)
during_result=np.array(during_result)

before_result=[]
for i in range(0,before_data.shape[0]):
    temp=[]
    for j in range(0,before_data.shape[1]):
        temp.append([1,0])
    before_result.append(temp)
before_result=np.array(before_result)


X_train_during, X_test_during, y_train_during, y_test_during = train_test_split(
during_data, during_result, test_size=0.3, random_state=0)

X_train_before, X_test_before, y_train_before, y_test_before = train_test_split(
before_data, before_result, test_size=0.3, random_state=0)


def fun(x1,x2):
    return np.append(x1,x2,axis=0)

X_train=fun(X_train_before,X_train_during)
y_train=fun(y_train_before,y_train_during)
X_test=fun(X_test_before,X_test_during)
y_test=fun(y_test_before,y_test_during)

print(X_test.shape)
print(y_test.shape)
training_data=[]
testing_data=[]

# a=[]
# b=[]
for i in range(0,len(X_train)):
    for j in range(0,30):
        training_data.append([X_train[i][j],y_train[i][j]])

for i in range(0,len(X_test)):
    for j in range(0,30):
        testing_data.append([X_test[i][j],y_test[i][j]])

# for i in training_data:
#     x=np.array(i)
    # print(x.shape)

# shuffle(training_data)
training_data=np.array(training_data)
testing_data=np.array(testing_data)

print(training_data.shape[0])
print(testing_data.shape[0])
np.save('data/train/training_data.npy',training_data)
np.save('data/test/testing_data.npy',testing_data)
