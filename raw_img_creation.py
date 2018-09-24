# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

#tesisHome = os.path.join('/', 'home', 'diegomartinez33', 'Documents', 'Uniandes', '2018-20')
tesisHome = os.path.join('/', 'hpcfs','home', 'da.martinez33', 'Tesis')

dataCreationfolder = os.path.join(tesisHome, 'Codes', 'dataCreation')
sys.path.append(dataCreationfolder)

import createData
import ExpDataCreation as expD

#ex_hdd_path = os.path.join('/', 'media', 'diegomartinez33', 'TOSHIBA EXT')

local_data_path = os.path.join(tesisHome,'Data', 'DOE_databases', 'balancedData')

raw_img_path = os.path.join(local_data_path, 'raw_imgs')

if not os.path.exists(raw_img_path):
    os.makedirs(raw_img_path)

print('BEGINNING RAW IMAGES CREATION\n\n')

for file in os.listdir(local_data_path):
    if file.endswith(".pkl"):
        file_path = os.path.join(local_data_path, file)
        print(file)
        print('Start loading...')
        baseDict = createData.loadData(file_path)
        print('Loading finished...')
        # print(baseDict.keys())

        FinalDict = {}
        print('---------- Start images creation... -------------')
        for fold in baseDict.keys():
            fold_dict = baseDict[fold]
            # print(fold_dict.keys())
            imgData = expD.getSigImages(fold_dict, ostype=1)
            foldDictImg = {}
            foldDictImg['images'] = imgData
            foldDictImg['instances'] = fold_dict['instances']
            foldDictImg['Targets'] = fold_dict['Targets']
            FinalDict[fold] = foldDictImg

        rawFileName = file[:16] + 'raw' + file[19:]
        savePath = os.path.join(raw_img_path, rawFileName)
        pickle.dump(FinalDict, open(savePath, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('Saved raw dataBase: ', rawFileName)

        print('---------- Finished images creation for file: ', file, ' ------------')
    # sys.exit()

print('\n\nFINISHED RAW IMAGES CREATION')


