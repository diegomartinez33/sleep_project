# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 17:26:35 2018

@author: da.martinez33
"""

import numpy as np
import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#dataCreationfolder = os.path.join('~','Tesis','Codes','dataCreation')
#sys.path.append(dataCreationfolder)
#from dataCreation import ExpDataCreation as expD
from . import ExpDataCreation as expD

homeDic = '/hpcfs/home/da.martinez33'

def createDataset(datasetNum=0, windowSize=30, rawImages=False, delOuts='No', 
                  timesOut=0, 
                  jobPathPrint=os.path.join(homeDic,'Tesis','results')):
    """ Function that creates the defined database
    
    datasetNum = 0
    0 for 'SC' 
    1 for 'ST'
    2 for 'SC_ST'
    
    windowSize = 30
    Can be 1s 5s 10s 15s or 30s
    
    If you want to get raw images from the signal, set True
    #else it will just create amplitude matrix for signals
    #rawImages = False or True
    
    """
        
    def delOutilers(data,desvTimes=2):
        """ Function to delete outliers from a database """
        dataflat = data.flatten()
        mean1 = np.mean(dataflat)
        std1 = np.std(dataflat)
        
        threshold = mean1 + (std1 * desvTimes)
        
        for i,val in enumerate(data):
            for j,val2 in enumerate(val):
                for k,val3 in enumerate(val2):
                    if val3 > threshold:
                        data[i,j,k] = threshold
                    elif val3 < threshold * -1:
                        data[i,j,k] = threshold * -1
                    else:
                        continue
        
        return data
        
    
    dataDir_3FCV = os.path.join(homeDic,'Tesis','Data','Data_3FCV')
    #dataSets = ['SC', 'ST', 'SC_ST']
    dataSets = ['SC', 'ST']
    
    dataType = dataSets[datasetNum]
    
    #def createDataset(datasetNum=0, windowSize=30, rawImages=False, delOuts='No', 
    #      timesOut=0):
    if rawImages:
        imgType = 'raw'
    else:
        imgType = 'amp'
    print_file_job = 'printJOB_{}_{}_{}_{}.txt'.format(dataType,imgType,
                               windowSize,timesOut)
    #job_print_dir = os.path.join(homeDic,'Tesis','Results',print_file_job)
    job_print_dir = os.path.join(jobPathPrint,print_file_job)
    templateLine = 'DataSet created: {}  Window Size: {:.0f} ' 
    templateLine += 'Images: {} Eliminate Outliers: {} timesSTD: {:3}\n'
    First_line = templateLine.format(dataType,
                                      windowSize, rawImages, delOuts, timesOut)
    
    if os.path.isfile(job_print_dir):
        with open(job_print_dir,'a') as f:
            f.write(First_line)
            print(First_line)
            f.close()
    else:
        with open(job_print_dir,'w') as f:
            f.write(First_line)
            print(First_line)
            f.close()
    
    FinalDict = {}
    
    # cont = 0
    dataPath = os.path.join(dataDir_3FCV,dataType) 
    for fold in os.listdir(dataPath):
        cont = 0
        foldPath = os.path.join(dataPath,fold)   
        print('Fold running: ', fold)
        for patient in os.listdir(foldPath):
            patientPath = os.path.join(foldPath, patient)
            print('Number of patient: ',patient)
            
            for file in os.listdir(patientPath):
                if file.endswith(".npz"):
                    filePath = os.path.join(patientPath,file)
                    dictData = np.load(filePath)
                    
                    #Divide signal in windows
                    windowedData = expD.sepWins(dictData,windowSize,dataType)
                                                           
                    # Catenated info
                    if cont > 0:
                        Data = np.append(Data,windowedData['data'],axis=1)
                        Data2 = np.append(Data2,windowedData['data2'],axis=1)
                        Targets = np.append(Targets,windowedData['targets'])
                    else:
                        Data = windowedData['data']
                        Data2 = windowedData['data2']
                        Targets = windowedData['targets']
                    cont += 1
            
                    line = 'Database: {} Fold: {}  Patient: {} ' 
                    line += 'Record: {}\n'
                    next_line = line.format(dataType,fold, patient, file)
                    with open(job_print_dir,'a') as f:
                        f.write(next_line)
                        print(next_line)
                        f.close()
        
        #Create dictionary per fold
        foldDict = {}
        if delOuts == 'Yes' and timesOut:
            foldDict['data'] = delOutilers(Data,timesOut)
        else: 
            foldDict['data'] = Data
            
        foldDict['data2'] = Data2
        foldDict['Targets'] = Targets
        
        #Get raw images
        if rawImages:
            image_line_ini = 'Start Creation of images for Fold: {}\n'
            image_line_ini = image_line_ini.format(fold)
            image_line_end = 'Finished Creation of images for Fold: {}\n'
            image_line_end = image_line_end.format(fold)

            with open(job_print_dir, 'a') as f:
                f.write(image_line_ini)
                print(image_line_ini)
                f.close()

            # Get images from data in foldDict
            imgData = expD.getSigImages(foldDict,ostype=1)
            foldDictImg = {}
            foldDictImg['images'] = imgData
            foldDictImg['Targets'] = Targets
            
            #Add image dictionary to big dictionary of data
            FinalDict[fold] = foldDictImg
            print('Fold created: ', fold)

            with open(job_print_dir, 'a') as f:
                f.write(image_line_end)
                print(image_line_end)
                f.close()
        else:
            #Add amplitude dictionary to big dictionary of data
            FinalDict[fold] = foldDict
            print('Fold created: ', fold)
    
    return FinalDict
    
def getHist(completeDict):
        """ Function to generate histograms of databases """
        k = completeDict.keys()
        for key in k:
            fdict = completeDict[key]
            data = fdict['data']
            data2 = fdict['data2']
            
            dataflat = data.flatten()
            data2flat = data2.flatten()
            
            
            mean1 = np.mean(dataflat)
            std1 = np.std(dataflat)
            mean2 = np.mean(data2flat)
            std2 = np.std(data2flat)
            
            print('Mean of ' + key + 'data :', mean1)
            print('STD of ' + key + 'data :', std1)
            print('Mean of ' + key + 'data_2 :', mean2)
            print('STD of ' + key + 'data_2 :', std2)
            
            numbins = 100
            
            
            #plt.hist(dataflat,numbins,(-300,300))
            #plt.savefig("hist_Data_" + key + ".png")
            
            #plt.cla()
            
            #plt.hist(data2flat,numbins,(-300,300))
            #plt.savefig("hist_Data_" + key + ".png")
            
            #plt.cla()
            
            
            plt.hist(dataflat,numbins,(dataflat.min(), dataflat.max()))
            plt.savefig("hist_Data_" + key + ".png")
            
            plt.cla()
            
            plt.hist(data2flat,numbins,(data2flat.min(), data2flat.max()))
            plt.savefig("hist_Data_2" + key + ".png")
            
            plt.cla()            
                
            

