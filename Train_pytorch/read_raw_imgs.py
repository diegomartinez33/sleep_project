import numpy as np
import os.path as osp
import sys

TesisPath = '/hpcfs/home/da.martinez33/Tesis'

data_base_path = osp.join(TesisPath, 'Data', 'DOE_databases', 'balancedData', 'raw_imgs')
data_base_file = 'dataBaseDict_SC_raw_10_0.pkl'
data_base_file_2 = 'dataBaseDict_SC_raw_30_3.pkl'

codesfolder = osp.join(TesisPath, 'Codes')
sys.path.append(codesfolder)

import createData

print('Loading data....\n')
databaseDict = createData.loadData(osp.join(data_base_path, data_base_file_2))
print('Data Loaded!\n')

fold_1 = databaseDict['FirstSC']
print(fold_1.keys())

data_imgs = fold_1['images']

print(data_imgs.shape)