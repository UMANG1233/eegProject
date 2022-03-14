from glob import glob
import numpy as np
import mne
import os
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

all_file_path=glob(r'C:\\Users\\goenk\\Desktop\\Machine-Learning\\eegArithmeticTask\\eeg-during-mental-arithmetic-tasks-1.0.0\\eeg-during-mental-arithmetic-tasks-1.0.0\\*.edf')

# seperating three minute and one minute data
before_file_path=[i for i in all_file_path if '1' in i.split('_')[1]]
during_file_path=[i for i in all_file_path if '2' in i.split('_')[1]]
before_file_path.sort()
during_file_path.sort()
ct=0

# Reading raw data from eeg files
def read_data(file_path,time,st,en):
  data= mne.io.read_raw_edf(file_path, preload=True)
  data.set_eeg_reference()
  data.filter(l_freq=0.5, h_freq=50, filter_length=time)
  if time=="180s":
    data = data.copy().crop(tmin=st, tmax=en)
  else: 
    data=data.copy().crop(tmin=st, tmax=en)
  epochs=mne.make_fixed_length_epochs(data, duration=2,overlap=0)
  array=epochs.get_data()
  return array

# Creating two seperate arrays for before and after
before_epochs_array=[]
during_epochs_array=[]
for i in before_file_path:
  if ct!=31:
    before_epochs_array.append(read_data(i,"180s",60,120))
  else: 
    before_epochs_array.append(read_data(i,"180s",10,70))
  ct=ct+1

for i in during_file_path:
  during_epochs_array.append(read_data(i,"60s",0,60))

before_epochs_array=np.array(before_epochs_array)
during_epochs_array=np.array(during_epochs_array)
# data is now 4 dimension=(no. of people, 60s/2,no. of channel, 500*2s)
# (36,30,21,1000)

np.save('data/before_file_path/before_data.npy',before_epochs_array)
np.save('data/during_file_path/during_data.npy',during_epochs_array)