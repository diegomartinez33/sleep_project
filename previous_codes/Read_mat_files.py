# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 23:37:52 2018

@author: diega
"""

#Código de creación de base de datos Tesis I

#Lectura de archivos .mat

#import h5py
import numpy as np
from numpy import ndarray
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import time

#matsdir = 'G:\Diego_Projects\DatabasePSGLab\PSGLab_datos\De_German\OriginalData'

os.chdir('C:\\Users\\da.martinez33\\OneDrive - Universidad de Los Andes\\UNIANDES\\UNIANDES 2018-10\\Tesis I\\Codes')

filename = 'SC4001E0.mat'
filename_2 = 'SC4001E0-PSG_annot.mat'
#%%
#mat = sio.loadmat(filename)
mat2 = sio.loadmat(filename_2)
#f = h5py.File(filename,'r')
#variables = mat.keys()

#data = mat["data"]
#targets = mat["Targets"]

target_2 = mat2["Anot_Cell"]

#pass target_2 from ndarray to list
labs = []
for a in target_2[1:]:
    auxlist = []
    for b in a:
        x = b[0][0]
        auxlist.append(x)
    
    labs.append(auxlist)

labs = np.array(labs)
print(labs.shape)

#data = np.array(data)
#targets = np.array(targets)

#%%
#
#listdata = np.reshape(data,data.shape[0]*data.shape[1])
#
#num_bins = 15
#n, bins, patches = plt.hist(listdata[0:15000], num_bins, facecolor='blue', alpha=0.5)
#plt.show()
##%%
#print(np.std(listdata))
#%%

