# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 12:03:19 2018

@author: da.martinez33
"""

#Código de lectura de etiquetas base de datos Tesis I

#Lectura de archivos .mat

#import h5py
#import sio
import numpy as np
import scipy.io as sio
#import matplotlib.pyplot as plt
import os

##%%
#os.chdir('C:\\Users\\da.martinez33\\OneDrive - Universidad de Los Andes\\UNIANDES\\UNIANDES 2018-10\\Tesis I\\Codes')
#
#filename_2 = 'SC4001E0-PSG_annot.mat'
#mat2 = sio.loadmat(filename_2)
##f = h5py.File(filename,'r')
##variables = mat2.keys()
#
#target_2 = mat2["Anot_Cell"]
##target_2 = np.array(target_2)

#%%
# Vectores con numero de carpetas de cada paciente
# SC
v_numSC = list(range(0,20))
v_numST = list(range(1,24))
del v_numST[22]
del v_numST[2]

# Directorios de bases de datos SC y ST
databaseDir_PSG_SC = os.path.join('..','Data','Organized','SC')
databaseDir_PSG_ST = os.path.join('..','Data','Organized','ST')

# Read mat files with annotations
def getLabels(matfile_path):
    
    mat2 = sio.loadmat(matfile_path)
    target_2 = mat2["Anot_Cell"]
    
    labs = []
    for a in target_2[1:]:
        auxlist = []
        for b in a:
            x = b[0][0]
            auxlist.append(x)
        
        labs.append(auxlist)
    
    labs = np.array(labs)
    return labs
    #print(labs.shape)
    
    #Create an array with labels each 30 seconds
    # maybe inside a function with variable window size

# SC data

for i in v_numSC:
    folderDir = os.path.join(databaseDir_PSG_SC,str(i))
    for file in os.listdir(folderDir):
        if file.endswith("annot.mat"):
            Targets = getLabels(os.path.join(folderDir,file))
            
            npz_file = file[0:12] + '.npz'
            #Get info from already edf readed files (saved in .npz)
            datafile = np.load(os.path.join(folderDir,npz_file))
            sigData = datafile["sigData"];
            labels = datafile["labels"];
            
            #save Targets in same file
            np.savez(os.path.join(folderDir,npz_file),labels=labels,
                     sigData=sigData,Targets=Targets)
            print(file)
        
for i in v_numST:
    folderDir = os.path.join(databaseDir_PSG_ST,str(i))
    for file in os.listdir(folderDir):
        if file.endswith("annot.mat"):
            Targets = getLabels(os.path.join(folderDir,file))
            
            npz_file = file[0:12] + '.npz'
            #Get info from already edf readed files (saved in .npz)
            datafile = np.load(os.path.join(folderDir,npz_file))
            sigData = datafile["sigData"];
            labels = datafile["labels"];
            
            #save Targets in same file
            np.savez(os.path.join(folderDir,npz_file),labels=labels,
                     sigData=sigData,Targets=Targets)
            print(file)
    
    
    
    