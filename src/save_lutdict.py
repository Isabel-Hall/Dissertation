from scipy.io import loadmat
import numpy as np
import mat73


dict_file = mat73.loadmat("/home/issie/data/my_matlab_dict/small_brain_dict_bSSFP_10tr.mat")
#print(type(dict_file)) #<class 'dict'>
#print(len(dict_file))
#print(list(dict_file))
# dictionary with keys: ['RFpulses', 'T1', 'T2', 'TR', 'dict', 'dict_norm', 'lut']

lut = dict_file["lut"]
print(type(lut)) #<class 'numpy.ndarray'>
print(len(lut))
np.save("/home/issie/data/dict_tables/lut.npy", lut)
dic = dict_file["dict"]
print(type(dic)) #<class 'numpy.ndarray'>
print(len(dic)) #1000
np.save("/home/issie/data/dict_tables/dic.npy", dic)
dic_norm = dict_file["dict_norm"]
np.save("/home/issie/data/dict_tables/dic_norm.npy", dic_norm)
