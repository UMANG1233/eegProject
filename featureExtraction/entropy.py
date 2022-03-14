import EntropyHub as EH
from cv2 import normalize
import numpy as np
import ordpy
from slope_calculation import fComputeSlopeEntropy

entropy_before_raw=np.load('../data/before_file_path/before_data.npy')
entropy_during_raw=np.load('../data/during_file_path/during_data.npy')

# all entropy features
dispersion_entropy=[]
approximation_entropy=[]
sample_entropy=[]
permutation_entropty=[]
slope_entropy=[]
m=2

# entropy functions 
def entropy_calc(x):
    dispersion=EH.DispEn(x)

    approximation=EH.ApEn(x)
    
    sample=EH.SampEn(x)
    
    # permutation=EH.PermEn(x) This is not giving proper results
    permutation=ordpy.permutation_entropy(x)
    
    slope=EH.SlopEn(x)
    # slope = fComputeSlopeEntropy(x,2,1,0.001)
    answer=[]
    answer.append(dispersion[0])
    answer.append(approximation[0][m])
    answer.append(sample[0][m])
    answer.append(permutation)
    answer.append(slope)
    return answer
    # return dispersion,approximation,sample,permutation,slope

# Storing features according to our requirements
def entropy_start(entropy_array):
    temp=[]
    # ct=0
    entropy_for_36_people=[]
    for i in entropy_array:
        entropy_for_30_epochs=[]
        for k in i:
            entropy_for_21_channel=[]
            for j in range(0,len(k)):
                x=k[j]
                entropy_for_1_channel=entropy_calc(x)
                entropy_for_21_channel.append(entropy_for_1_channel)
            entropy_for_30_epochs.append(entropy_for_21_channel)
            # ct=ct+1
            # print(ct)
        entropy_for_36_people.append(entropy_for_30_epochs)
    return entropy_for_36_people



entropy_before=np.array(entropy_start(entropy_before_raw))
entropy_during=np.array(entropy_start(entropy_during_raw))
# slope_entropy=np.array(slope_entropy)
# slope_entropy=slope_entropy.astype(np.float256)
print(entropy_before.shape)
print(entropy_during.shape)
# print(np.array(dispersion_entropy).shape)
# print(np.array(approximation_entropy).shape)
# print(np.array(sample_entropy).shape)
# print(np.array(permutation_entropty).shape)
# print(np.array(slope_entropy).shape)
# print(dispersion_entropy)
# print(approximation_entropy)
# print(sample_entropy)
# print(permutation_entropty)
# print(slope_entropy)

np.save('../data/during_file_path/entropy_during.npy',entropy_during)
np.save('../data/before_file_path/entropy_before.npy',entropy_before)
# np.save('../data/during_file_path/sampleEntropy.npy',sample_entropy)
# np.save('../data/during_file_path/permutationEntropy.npy',permutation_entropty)
# np.save('../data/during_file_path/slopeEntropy.npy',slope_entropy)