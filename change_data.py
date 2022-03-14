import numpy as np

training_data=np.load('data/train/training_data.npy',allow_pickle=True)

for i in range(0,1500):
    # x=i[0]
    # print(x.shape)
    for j in range(4,105,5):
        training_data[i][0][j]=0
    # print(i[0][4])

# for i in range(0,1500):
#     print(training_data[i][0])
np.save('data/train/training_data_change.npy',training_data)