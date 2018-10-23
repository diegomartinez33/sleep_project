# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import os.path as osp

TesisPath = '/hpcfs/home/da.martinez33/Tesis'

not_balanced_path = osp.join(TesisPath, 'Data', 'DOE_databases')
balanced_path = osp.join(TesisPath, 'Data', 'DOE_databases', 'balancedData')

files = ['dataBaseDict_SC_amp_10_0.pkl', 'dataBaseDict_SC_amp_30_0.pkl']
files = ['dataBaseDict_ST_amp_10_0.pkl', 'dataBaseDict_ST_amp_30_0.pkl']


print('---------------- NOT BALANCED DATA -------------------')
for file in files:
    print('\n\nFile: ', file)
    file_path = osp.join(not_balanced_path,file)
    database = pickle.load(open(file_path,'rb'))
    if file[-15:-13] == 'SC':
        fold_1 = database['FirstSC']['Targets']
        fold_2 = database['SecondSC']['Targets']
        fold_3 = database['ThirdSC']['Targets']
    else:
        fold_1 = database['FirstST']['Targets']
        fold_2 = database['SecondST']['Targets']
        fold_3 = database['ThirdST']['Targets']
    
    for inst in range(0,7):
        cont = 0
        for i in fold_1:
            if i == inst:
                cont += 1
        print('Fold 1 --- ', 'Class: ', inst, 'instances: ', cont)
        cont = 0

        for i in fold_2:
            if i == inst:
                cont += 1
        print('Fold 2 --- ', 'Class: ', inst, 'instances: ', cont)
        cont = 0

        for i in fold_3:
            if i == inst:
                cont += 1
        print('Fold 3 --- ', 'Class: ', inst, 'instances: ', cont)
        cont = 0

print('\n\n\n')
print('---------------- BALANCED DATA -------------------')
for file in files:
    print('\n\nFile: ', file)
    file_path = osp.join(balanced_path,file)
    database = pickle.load(open(file_path,'rb'))

    if file[-15:-13] == 'SC':
        fold_1 = database['FirstSC']['Targets']
        fold_2 = database['SecondSC']['Targets']
        fold_3 = database['ThirdSC']['Targets']
    else:
        fold_1 = database['FirstST']['Targets']
        fold_2 = database['SecondST']['Targets']
        fold_3 = database['ThirdST']['Targets']

    for inst in range(0,7):
        cont = 0
        for i in fold_1:
            if i == inst:
                cont += 1
        print('Fold 1 --- ', 'Class: ', inst, 'instances: ', cont)
        cont = 0

        for i in fold_2:
            if i == inst:
                cont += 1
        print('Fold 2 --- ', 'Class: ', inst, 'instances: ', cont)
        cont = 0

        for i in fold_3:
            if i == inst:
                cont += 1
        print('Fold 3 --- ', 'Class: ', inst, 'instances: ', cont)
        cont = 0



