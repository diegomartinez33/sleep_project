# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:30:52 2018

@author: da.martinez33
"""

# Test dataCreation

import sys
import os
import numpy as np
import scipy

dataCreationfolder = os.path.join('~','Tesis','Codes')
sys.path.append(dataCreationfolder)

homeDic = '/hpcfs/home/da.martinez33'

dataPath = os.path.join(homeDic,'Tesis','Data','DOE_databases')
dataFile = 'dataBaseDict_SC_amp_30_0.pkl'

import createData


#createData.makeData(datasetNum=0,windowSize=30,rawImages=False,delOuts='No',
#             timesOut=0,hists='No')

createData.balanceData(dataPath,dataFile)