# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 19:34:28 2018

@author: diega
"""
import numpy as np
import scipy.io as sio
import os
from shutil import copyfile
import time

homeDic = '/hpcfs/home/da.martinez33'

## Create 3 fold Cross-Validation database
dataDir_3FCV = os.path.join('..','Data','Data_3FCV')
if not os.path.exists(dataDir_3FCV):
    os.makedirs(dataDir_3FCV)

# Directorios de bases de datos SC y ST
databaseDir_PSG_SC = os.path.join(homeDic,'Tesis','Data','Organized','SC')
databaseDir_PSG_ST = os.path.join(homeDic,'Tesis','Data','Organized','ST')
#%%
# Vectores con numero de carpetas de cada paciente
# SC
v_numSC = list(range(0,20))
v_numST = list(range(1,24))
del v_numST[22]
del v_numST[2]

fold_size = round(len(v_numSC) * 1/3)

randSC = np.random.permutation(v_numSC)
randST = np.random.permutation(v_numST)

FirstSC = randSC[0:fold_size]
SecondSC = randSC[fold_size:fold_size*2]
ThirdSC = randSC[fold_size*2:]

FirstST = randST[0:fold_size]
SecondST = randST[fold_size:fold_size*2]
ThirdST = randST[fold_size*2:]

print("---------------------- For SC data ---------------")
print('FirstSC:',FirstSC)
print('SecondSC:',SecondSC) 
print('ThirdSC:',ThirdSC)  

print("---------------------- For ST data ---------------")
print('FirstST:',FirstST)
print('SecondST:',SecondST) 
print('ThirdST:',ThirdST)  

#%%
print("------------------- Start copying ----------------")
############## SC ################### 
print("------------------- SC only ----------------")
## First Folder
print('First Folder:')
First_folder = os.path.join(dataDir_3FCV,'SC','FirstSC')
if not os.path.exists(First_folder):
    os.makedirs(First_folder)
    print(First_folder)

#Patients subfolders
for i in FirstSC:
    folder = os.path.join(First_folder,str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderSC = os.path.join(databaseDir_PSG_SC,str(i))
    for file in os.listdir(folderSC):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderSC,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

## Second Folder
print('Second Folder:')
Second_folder = os.path.join(dataDir_3FCV,'SC','SecondSC')
if not os.path.exists(Second_folder):
    os.makedirs(Second_folder)
    print(Second_folder)
    
#Patients subfolders
for i in SecondSC:
    folder = os.path.join(Second_folder,str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderSC = os.path.join(databaseDir_PSG_SC,str(i))
    for file in os.listdir(folderSC):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderSC,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

## Third Folder
print('Third Folder:')
Third_folder = os.path.join(dataDir_3FCV,'SC','ThirdSC')
if not os.path.exists(Third_folder):
    os.makedirs(Third_folder)
    print(Third_folder)

#Patients subfolders
for i in ThirdSC:
    folder = os.path.join(Third_folder,str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderSC = os.path.join(databaseDir_PSG_SC,str(i))
    for file in os.listdir(folderSC):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderSC,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

print("------------------- SC Finished ----------------")

time.sleep(10)

############## ST ###################
print("------------------- ST only ----------------")
## First Folder
print('First Folder:')
First_folder = os.path.join(dataDir_3FCV,'ST','FirstST')
if not os.path.exists(First_folder):
    os.makedirs(First_folder)
    print(First_folder)

#Patients subfolders
for i in FirstST:
    folder = os.path.join(First_folder,str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderST = os.path.join(databaseDir_PSG_ST,str(i))
    for file in os.listdir(folderST):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderST,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

## Second Folder
print('Second Folder:')
Second_folder = os.path.join(dataDir_3FCV,'ST','SecondST')
if not os.path.exists(Second_folder):
    os.makedirs(Second_folder)
    print(Second_folder)

#Patients subfolders
for i in SecondST:
    folder = os.path.join(Second_folder,str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderST = os.path.join(databaseDir_PSG_ST,str(i))
    for file in os.listdir(folderST):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderST,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

## Third Folder
print('Third Folder:')
Third_folder = os.path.join(dataDir_3FCV,'ST','ThirdST')
if not os.path.exists(Third_folder):
    os.makedirs(Third_folder)
    print(Third_folder)

#Patients subfolders
for i in ThirdST:
    folder = os.path.join(Third_folder,str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderST = os.path.join(databaseDir_PSG_ST,str(i))
    for file in os.listdir(folderST):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderST,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)
            
print("------------------- ST Finished ----------------")

time.sleep(10)

############## SC & ST ###################
print("------------------- SC and ST ----------------")

FirstST_2 = np.array(FirstST) + 20
SecondST_2 = np.array(SecondST) + 20
ThirdST_2 = np.array(ThirdST) + 20

FirstST_2 = list(FirstST_2)
SecondST_2 = list(SecondST_2)
ThirdST_2 = list(ThirdST_2)

FirstSCST = list(FirstSC) + FirstST_2
SecondSCST = list(SecondSC) + SecondST_2
ThirdSCST = list(ThirdSC) + ThirdST_2
## First Folder
print('First Folder:')
First_folder = os.path.join(dataDir_3FCV,'SC_ST','First')
if not os.path.exists(First_folder):
    os.makedirs(First_folder)
    print(First_folder)

#Patients subfolders

for i in FirstSC:
    folder = os.path.join(First_folder,str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderSC = os.path.join(databaseDir_PSG_SC,str(i))
    for file in os.listdir(folderSC):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderSC,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

for i in FirstST:
    folder = os.path.join(First_folder,str(i + 20))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderST = os.path.join(databaseDir_PSG_ST,str(i))
    for file in os.listdir(folderST):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderST,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

## Second Folder
print('Second Folder:')
Second_folder = os.path.join(dataDir_3FCV,'SC_ST','Second')
if not os.path.exists(Second_folder):
    os.makedirs(Second_folder)
    print(Second_folder)

#Patients subfolders

for i in SecondSC:
    folder = os.path.join(Second_folder,str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderSC = os.path.join(databaseDir_PSG_SC,str(i))
    for file in os.listdir(folderSC):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderSC,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

for i in SecondST:
    folder = os.path.join(Second_folder,str(i + 20))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderST = os.path.join(databaseDir_PSG_ST,str(i))
    for file in os.listdir(folderST):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderST,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

## Third Folder
print('Third Folder:')
Third_folder = os.path.join(dataDir_3FCV,'SC_ST','Third')
if not os.path.exists(Third_folder):
    os.makedirs(Third_folder)
    print(Third_folder)

#Patients subfolders

for i in ThirdSC:
    folder = os.path.join(Third_folder,str(i))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderSC = os.path.join(databaseDir_PSG_SC,str(i))
    for file in os.listdir(folderSC):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderSC,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)

for i in ThirdST:
    folder = os.path.join(Third_folder,str(i + 20))
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folderST = os.path.join(databaseDir_PSG_ST,str(i))
    for file in os.listdir(folderST):
        bool1 = file.endswith("PSG.edf")
        bool2 = file.endswith("annot.mat")
        bool3 = file.endswith("PSG.npz")
        if bool1 or bool2 or bool3:
            file_src = os.path.join(folderST,file)
            folder_dst = os.path.join(folder,file)
            copyfile(file_src,folder_dst)
            print(file)
            
print("------------------- SC_ST Finished ----------------")

print("---------------------- For SC data ---------------")
print('FirstSC:',FirstSC)
print('SecondSC:',SecondSC) 
print('ThirdSC:',ThirdSC)  

print("---------------------- For ST data ---------------")
print('FirstST:',FirstST)
print('SecondST:',SecondST) 
print('ThirdST:',ThirdST)

print("---------------------- For SC and ST data ---------------")

print('FirstST:',FirstSCST)
print('SecondST:',SecondSCST) 
print('ThirdST:',ThirdSCST)