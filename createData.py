    # -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 09:50:24 2018

@author: da.martinez33
"""
import numpy as np
import sys
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# Code to make Factorial Experiments
dataCreationfolder = os.path.join('~','Tesis','Codes','dataCreation')
sys.path.append(dataCreationfolder)

from dataCreation import ExpDataCreation
from dataCreation import getDatabase as gD

#datasetNum, windowSize, rawImages = 0, 30, False
#delOuts='No'
#timesOut=0
homeDic = '/hpcfs/home/da.martinez33'
saveDataPath = os.path.join(homeDic,'Tesis','Data','DOE_databases')

def makeData(datasetNum=0,windowSize=30,rawImages=False,delOuts='No',
             timesOut=0,savePath=saveDataPath,jobPathPrint=os.path.join(homeDic,'Tesis','results'),
             hists='No'):
    """ Function to create database dictionary based on parameters of type of
    database in physionet, window size, amplitude or rawImages, eliminate or
    not the outliers, times of STD to be truncated
    
    Its also posible to input the path in which progess is shown when running
    a job in the cluster; the path which database dictionaries will be saved
    and tha possibility to create histograms for the database
    
    """
    startTime = time.time()
    dataBaseDict = gD.createDataset(datasetNum,windowSize,rawImages,delOuts,
                                    timesOut,jobPathPrint)
    endTime = time.time()
    
    runningTime = endTime - startTime
    print('Total minutes used to create the data base: ', runningTime/60)
    
    dataSets = ['SC', 'ST']
    dataType = dataSets[datasetNum]
    
    if rawImages:
        imgType = 'raw'
    else:
        imgType = 'amp'
    dataBaseName = 'dataBaseDict_{}_{}_{}_{}.pkl'
    dataBasefile = dataBaseName.format(dataType,imgType,windowSize,timesOut)
    dataSavePath = os.path.join(savePath,dataBasefile)
    
    pickle.dump( dataBaseDict, open(dataSavePath, 'wb'), 
                pickle.HIGHEST_PROTOCOL)
    print('Saved dataBase')
    
    if hists == 'Yes':
        gD.getHist(dataBaseDict)      
    return dataBaseDict

def loadData(dataPathFile):
    """ Function to load the database dictionary from input path of pickle """
    if dataPathFile[-3:] == 'pkl':
        dataBaseDict = pickle.load(open(dataPathFile, 'rb'))
        return dataBaseDict
    else:
        raise Exception('File that is trying to be loaded is not a pickle file\n')
    
def balanceData(dataPath,dataPathFile):
    """ Function to make a balanced database"""
    
    databaseDict = loadData(os.path.join(dataPath,dataPathFile))
    classes = 7
    balancedDict = {}
    
    print('Start Balancing data')
    for fold in databaseDict.keys():
        fold_dic_not_balanced = databaseDict[fold]
        targetsVec = fold_dic_not_balanced['Targets']
        classesList = [[] for i in range(classes)]
        
        foldKeys = fold_dic_not_balanced.keys()
        print(foldKeys)
        
        if list(foldKeys)[0] == 'images':
            data = fold_dic_not_balanced['images']
        else:
            data = fold_dic_not_balanced['data']
            data2 = fold_dic_not_balanced['data2']
        
        
        cont = 0
        indx = 0
        for tgt in targetsVec:
            classesList[int(tgt)].append(indx)
            indx += 1
    
        instances = []
        for classinst in classesList:
            instances.append(len(classinst))
        
        instances = np.array(instances)
        org_instances = np.sort(instances)
        
        print(org_instances)
        
        min_inst = org_instances[1]
        print(min_inst)
        time.sleep(5)
        
        chann = 0
        for classinst in classesList:
            shuffledInst = np.array(classinst)
            np.random.shuffle(shuffledInst)
            cont2 = 0
            for j in shuffledInst[:min_inst]:
                aux_data = np.expand_dims(data[:,j,:],axis=1)
                aux_targ = targetsVec[j]
                if cont > 0:
                    balancedData = np.append(balancedData,aux_data,axis=1)
                    balancedTargets = np.append(balancedTargets,aux_targ)
                    if list(foldKeys)[0] != 'images':
                        balancedData_2 = np.append(balancedData_2,
                                                   np.expand_dims(data2[:,j,:],
                                                               axis=1))
                else:
                    balancedData = aux_data
                    balancedTargets = aux_targ
                    if list(foldKeys)[0] != 'images':
                        balancedData_2 = np.expand_dims(data2[:,j,:],axis=1)
                cont += 1
                cont2 += 1
                print('Fold: ',fold, 'Channel: ',chann, 'Image num: ',cont2)
            chann += 1

        foldDict = {}
        foldDict['Targets'] = balancedTargets
        foldDict['instances'] = org_instances
        if list(foldKeys)[0] == 'images':
            foldDict['images'] = balancedData
        else:
            foldDict['data'] = balancedData
            foldDict['data2'] = balancedData_2
        
        balancedDict[fold] = foldDict
    

    savePath = os.path.join(dataPath,'balancedData',dataPathFile)
    pickle.dump(balancedDict, open(savePath, 'wb'), pickle.HIGHEST_PROTOCOL)
    print('Saved Balanced dataBase')    
    
    return balancedDict
        
            
        
        #for i in range(len(classesList)):
        #    print(i)
    


