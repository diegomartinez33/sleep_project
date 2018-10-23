# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 09:39:47 2018

@author: da.martinez33
"""

import sys
import os
import numpy as np
import scipy

dataCreationfolder = os.path.join('~','Tesis','Codes')
sys.path.append(dataCreationfolder)

import createData

homeDic = '/hpcfs/home/da.martinez33'
saveDataPath = os.path.join(homeDic,'Tesis','Data','DOE_databases')
saveDataPath_PP = os.path.join(homeDic,'Tesis','Data','Per_Patient')

numFactors = 4
#combFactors = ff2n(numFactors)

Factors = {}
FactorsKeys = ['Data Type','Image type','Window size','Times STD outliers']
# Factors['Data Type'] = ['SC', 'ST']
Factors['Data Type'] = ['ST']
# Factors['Image type'] = ['Amplitude','Raw']
Factors['Image type'] = ['Amplitude']
Factors['Window size'] = [10,30]
Factors['Times STD outliers'] = [0,3]
# The second dimension is used to assign a coded level to every factor

followJobfile = 'dataCreationfile.txt'

#run = 0
#TotalRuns = 2 ** numFactors
def runData(histsDes='No',savePath=saveDataPath,
            jobPath=os.path.join(homeDic,'Tesis','results')):
    """ Function to create all combinations of databases"""
    run = 0
    TotalRuns = 2 ** numFactors
    for imgType in Factors['Image type']:
        if imgType == 'Amplitude':
            rawimages = False
        else:
            rawimages = True
        for dataType in Factors['Data Type']:
            if dataType == 'SC':
                datanum = 0
            else:
                datanum = 1
            for winSize in Factors['Window size']:
                for timesSTD in Factors['Times STD outliers']:
                    if timesSTD > 0:
                        deleteOuts = 'Yes'
                    else:
                        deleteOuts = 'No'
                    createData.makeData_PP(datasetNum=datanum,
                                        windowSize=winSize,
                                        rawImages=rawimages,
                                        delOuts=deleteOuts,
                                        timesOut=timesSTD,
                                        savePath=saveDataPath_PP,
                                        jobPathPrint=jobPath,
                                        hists=histsDes)
                    run += 1
                    print('Remaining runs: ',TotalRuns - run)
                    
                    line = 'Image Type: {} Database: {}  Window Size: {} ' 
                    line += 'Times STD out: {}\n'
                    next_line = line.format(imgType,dataType, winSize, timesSTD)
                    if os.path.isfile(os.path.join(jobPath,followJobfile)):
                        with open(os.path.join(jobPath,followJobfile),'a') as f:
                            f.write(next_line)
                            print(next_line)
                            f.close()
                    else:
                        with open(os.path.join(jobPath,followJobfile),'w') as f:
                            f.write(next_line)
                            print(next_line)
                            f.close()

if __name__ == "__main__":
    
    try:
        if str(sys.argv[1]) == 'hists':
            runData(histsDes='Yes')
        elif str(sys.argv[1]) == 'job_print_folder':
            pathJob = str(sys.argv[2])
            runData(jobPath=pathJob)
        elif str(sys.argv[1]) == 'save_folder':
            pathSave = str(sys.argv[2])
            runData(savePath=pathSave)
        else:
            runData()
    except:
        runData()

                