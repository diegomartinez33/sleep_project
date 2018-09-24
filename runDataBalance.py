# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:35:51 2018

@author: da.martinez33
"""

import sys
import os

dataCreationfolder = os.path.join('~','Tesis','Codes')
sys.path.append(dataCreationfolder)

import createData

homeDic = '/hpcfs/home/da.martinez33'

dataPath = os.path.join(homeDic,'Tesis','Data','DOE_databases')
jobFile = 'jobBalanceFollow.txt'
jobPath = os.path.join(homeDic,'Tesis','results')


for file in os.listdir(dataPath):
    if file.endswith(".pkl"):
        createData.balanceData(dataPath,file)
        print(file)

        if os.path.isfile(os.path.join(jobPath, jobFile)):
            with open(os.path.join(jobPath, jobFile), 'a') as f:
                f.write(file + '\n')
                f.close()
        else:
            with open(os.path.join(jobPath, jobFile), 'w') as f:
                f.write(file + '\n')
                f.close()
