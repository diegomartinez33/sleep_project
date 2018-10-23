# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

# Code to make Factorial Experiments
dataCreationfolder = os.path.join('~','Tesis','Codes')
sys.path.append(dataCreationfolder)

from dataCreation import ExpDataCreation
from dataCreation import getDatabase as gD

#datasetNum, windowSize, rawImages = 0, 30, False
#delOuts='No'
#timesOut=0
homeDic = '/hpcfs/home/da.martinez33'

dataPath = os.path.join(homeDic,'Tesis','Data','DOE_databases')
jobFile = 'jobAugmentationFollow.txt'
jobPath = os.path.join(homeDic,'Tesis','results')


saveDataPath = os.path.join(homeDic,'Tesis','Data','DOE_databases')

def augment(dataDict,winTime, sampling, dataType='SC'):
    """ Function to make subsampling of signals epochs to augment data
    
    dataDict must be a dictionary with data ('data' and 'data2') sampled 
    by 30 s windows and its targets ('targets'). If you are using data divided
    in 3 folds, you must input each fold ('FirstSC' or 'FirstST', etc...) 
    separately
    
    data shape (channels,wins,sizeWins) = (3,2650, 3000) or (4, 2650, 30)
    winTime is the length of window selected (5s,10s,15s) and must be a divisor 
    of 30 s
    Size of sampling (sampSize) in seconds (1s, 0,5s, etc...)
    
    """
    
    #labels_2_augment = [1,3,4,6]
    # sampling = 5
    
    #Info from 'data' key of dictionary dataDict
    Fs_1 = 100
    data1 = dataDict['data']
    
    #Info from 'data2' key of dictionary dataDict
    if dataType == 'ST':
        Fs_2 = 10
        labels_2_augment = [0,1,3,4,6]
    else:
        Fs_2 = 1
        labels_2_augment = [1,3,4,6]
    data2 = dataDict['data2']
    
    targets = dataDict['Targets']
    
    if winTime == 30:
        return dataDict
    else:
        def data_per_class(data):
            """ Function to obtain data instances per class """
            
            #% Classes:
            #% awake = 'W' -----------------> 0
            #% NREM Sleep Stage 1 = '1' ----> 1
            #% NREM Sleep Stage 2 = '2' ----> 2
            #% NREM Sleep Stage 3 = '3' ----> 3
            #% NREM Sleep Stage 4 = '4' ----> 4
            #% REM Sleep Stage = 'R' -------> 5
            #% Movement_time = 'e','?'------> 6
            classDataList = [[] for i in range(7)]
            #if dataType == 'ST':
            #    emptyArray = np.empty((4,1,winTime*Fs))
            #else:
            #    emptyArray = np.empty((3,1,winTime*Fs))
            
            #for i in range(7):
            #    classDataList[i] = [emptyArray, np.empty(1)]
            
            for inst in range(targets.shape[0]):
                label = targets[inst]
                new_data = data[:,inst,:]
                new_data = np.expand_dims(new_data, axis=1)
                if len(classDataList[label]) == 0:
                    classDataList[label] = [new_data, np.array(inst),
                    np.array(label)]
                else:
                    dataTarget = classDataList[label][0]
                    data_position = classDataList[label][1]
                    data_label = classDataList[label][2]
                    dataTarget = np.append(dataTarget, new_data, axis=1)
                    data_position = np.append(data_position,inst)
                    data_label = np.append(data_label,label)

                    classDataList[label][0] = dataTarget
                    classDataList[label][1] = data_position
                    classDataList[label][2] = data_label
            
            return classDataList
        
        def cutWindows(Data,instArray,sampSize,Fs,target):
            """ Function that make the cuts of sampling for all windows """
            movTime = sampSize * Fs
            winSize = winTime * Fs
            channels = Data.shape[0]
            finalData = np.empty((channels,1,winSize))
            finalTargets = np.empty((1))
            finalPosition = np.empty((1,2)) # first dim: position in orginal
                                               # data
                                            # second dim: position inside instance
            # after augmentation
            samp_count = 0
            
            for i in range(Data.shape[1]):
                origWin = Data[:,i,:]
                tar = target
                num_sampled_imgs = 0
                if tar in labels_2_augment:
                    aux1 = 0
                    aux2 = winSize
                    while aux2 <= origWin.shape[1]:
                        sampWin = origWin[:,aux1:aux2]
                        sampWin = np.expand_dims(sampWin,axis=1)
                        poscoord = np.expand_dims(np.array([instArray[i],
                                num_sampled_imgs]),axis=0)
                        if samp_count == 0:
                            finalData = sampWin
                            finalTargets = np.array(target)
                            finalPosition = poscoord
                        else:
                            finalData = np.append(finalData,sampWin,axis=1)
                            finalTargets = np.append(finalTargets,target)
                            finalPosition = np.append(finalPosition,poscoord,
                                axis=0)
                        aux1 += movTime
                        aux2 += movTime
                        num_sampled_imgs += 1
                        samp_count += 1

                else:
                    timesWin_in_dataWin = int(Data.shape[2]/winSize)
                    reshape_data = np.reshape(origWin,(channels,
                        timesWin_in_dataWin,winSize))
                    if samp_count == 0:
                        finalData = reshape_data
                        finalTargets = np.array(target)
                        for j in range(timesWin_in_dataWin-1):
                            finalTargets = np.append(finalTargets,target)
                    else:
                        finalData = np.append(finalData,reshape_data,axis=1)
                        for j in range(timesWin_in_dataWin):
                            finalTargets = np.append(finalTargets,target)
                    
                    num_sampled_imgs += 1
                    samp_count += 1

                # print(i)
            
            dataSampled = {}
            dataSampled['data'] = finalData
            dataSampled['targets'] = finalTargets
            dataSampled['position'] = finalPosition
            return dataSampled
                
        per_class_data1 = data_per_class(data1)
        #print(len(per_class_data1))
        #for k in range(7):
        #    print(per_class_data1[k][0].shape)
        
        #per_class_data2 = data_per_class(data2)
        for lab in range(len(per_class_data1)):
            
            data1 = per_class_data1[lab][0]
            data_pos = per_class_data1[lab][1] #Usar luego para LSTM
            
            # For data 1
            sampData1 = cutWindows(data1,data_pos,sampling,Fs_1,lab)
            per_class_data1[lab][0] = sampData1['data']
            per_class_data1[lab][1] = sampData1['position']
            per_class_data1[lab][2] = sampData1['targets']
                
            #For data 2
            #sampData2 = cutWindows(data2,Fs_2)
            
            print(sampData1['data'].shape)
            print(sampData1['targets'].shape)
            print(sampData1['position'].shape)

        #for k in range(7):
        #    print(per_class_data1[k][0].shape)
        
        #Catenation of all instances
        total_intances = 0
        for lab in range(len(per_class_data1)):
            if lab == 0:
                total_data = per_class_data1[lab][0]
                targets = per_class_data1[lab][2]
                total_intances += targets.shape[0]
                total_targets = targets
            else:
                #print(total_data.shape)
                #print(per_class_data1[lab][0].shape)
                total_data = np.append(total_data, per_class_data1[lab][0], axis=1)
                total_targets = np.append(total_targets,per_class_data1[lab][2])
    
    print(total_data.shape)
    print(total_targets.shape)
    ###########################################################################
    #sys.exit()
    ###########################################################################
    
    #Dictionary to save all cuted and oversampled data
    # Keys = ['data', 'data2', 'targets']
    FinalSampData = {}
    FinalSampData['data'] = total_data
    #FinalSampData['data2'] = sampData2['data']
    FinalSampData['targets'] = total_targets
    return FinalSampData

#def create_data_per_patient():
# TODO - create code to generate a database per patient