
import numpy as np
import os
import os.path as osp
import sys

import torch
import torch.utils.data as data_utils

TesisPath = '/hpcfs/home/da.martinez33/Tesis'

codesfolder = osp.join(TesisPath, 'Codes')
sys.path.append(codesfolder)

import createData

data_base_path = osp.join(TesisPath, 'Data', 'DOE_databases', 'balancedData')
data_base_file = 'dataBaseDict_SC_amp_10_0.pkl'

def loadDataBase(args, data_path=data_base_path,data_file=data_base_file,trainFold=1, raw=False, elim_artif=False,
                 joint_3_4=False):
    global train_loader
    global val_loader
    global test_loader

    databaseDict = createData.loadData(osp.join(data_path, data_file))

    if data_file[13:15] == 'SC':
        Fold1 = databaseDict['FirstSC']
        Fold2 = databaseDict['SecondSC']
        Fold3 = databaseDict['ThirdSC']
    else:
        Fold1 = databaseDict['FirstST']
        Fold2 = databaseDict['SecondST']
        Fold3 = databaseDict['ThirdST']


    #Verify is data is amplitud or raw plot
    if raw == False:
        data_fold1 = Fold1['data']
        data_fold2 = Fold2['data']
        data_fold3 = Fold3['data']
    else:
        data_fold1 = Fold1['images']
        data_fold2 = Fold2['images']
        data_fold3 = Fold3['images']

    label_fold1 = Fold1['Targets']
    label_fold2 = Fold2['Targets']
    label_fold3 = Fold3['Targets']

    if elim_artif == True:
        artif_label = 6
        instance_dim = 1

        list_fold1 = []
        for idx, lab in enumerate(label_fold1):
            if lab == artif_label:
                list_fold1.append(idx)

        list_fold2 = []
        for idx, lab in enumerate(label_fold2):
            if lab == artif_label:
                list_fold2.append(idx)

        list_fold3 = []
        for idx, lab in enumerate(label_fold3):
            if lab == artif_label:
                list_fold3.append(idx)

        data_fold1 = np.delete(data_fold1, tuple(list_fold1), instance_dim)
        label_fold1 = np.delete(label_fold1, tuple(list_fold1), 0)

        data_fold2 = np.delete(data_fold2, tuple(list_fold2), instance_dim)
        label_fold2 = np.delete(label_fold2, tuple(list_fold2), 0)

        data_fold3 = np.delete(data_fold3, tuple(list_fold3), instance_dim)
        label_fold3 = np.delete(label_fold3, tuple(list_fold3), 0)

    # if joint_3_4 == True:
    #     label_stage_3 = 3
    #     label_stage_4 = 4
    #
    #     for idx, lab in enumerate(label_fold1):
    #         if lab == label_stage_4:
    #             label_fold1[idx] = label_stage_3
    #
    #     for idx, lab in enumerate(label_fold2):
    #         if lab == label_stage_4:
    #             label_fold2[idx] = label_stage_3
    #
    #     for idx, lab in enumerate(label_fold3):
    #         if lab == label_stage_4:
    #             label_fold3[idx] = label_stage_3


    # train with 1 sets

    if trainFold == 1:
        # 1-fold
        data_train = data_fold1
        label_train = label_fold1
        data_val = data_fold2
        label_val = label_fold2
        data_test = data_fold3
        label_test = label_fold3
    elif trainFold == 2:
        # 2-fold
        data_train = data_fold2
        label_train = label_fold2
        data_val = data_fold3
        label_val = label_fold3
        data_test = data_fold1
        label_test = label_fold1
    else:
        # 3-fold
        data_train = data_fold3
        label_train = label_fold3
        data_val = data_fold1
        label_val = label_fold1
        data_test = data_fold2
        label_test = label_fold2

    # Organize data
    if raw == False:
        data_train = np.transpose(data_train, (1, 2, 0))
        data_val = np.transpose(data_val, (1, 2, 0))
        data_test = np.transpose(data_test, (1, 2, 0))

    data_train = np.expand_dims(data_train, axis=1)
    data_val = np.expand_dims(data_val, axis=1)
    data_test = np.expand_dims(data_test, axis=1)

    # Organize labels
    label_train = np.expand_dims(label_train, axis=1)
    label_train = label_train.astype(np.int64)

    label_val = np.expand_dims(label_val, axis=1)
    label_val = label_val.astype(np.int64)

    label_test = np.expand_dims(label_test, axis=1)
    label_test = label_test.astype(np.int64)

    print(data_train.shape)
    print(label_train.shape)
    print(data_val.shape)
    print(label_val.shape)
    print(data_test.shape)
    print(label_test.shape)
    # GPS

    # adjust train data
    tdata = torch.from_numpy(data_train)
    tlabel = torch.from_numpy(label_train)
    train = data_utils.TensorDataset(tdata, tlabel)
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    # adjust val data
    valdata = torch.from_numpy(data_val)
    vallabel = torch.from_numpy(label_val)
    val = data_utils.TensorDataset(valdata, vallabel)
    val_loader = data_utils.DataLoader(val, batch_size=args.batch_size, shuffle=True)

    # adjust test data
    testdata = torch.from_numpy(data_test)
    testlabel = torch.from_numpy(label_test)
    test = data_utils.TensorDataset(testdata, testlabel)
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=True)

    loaders = (train_loader,val_loader,test_loader)
    return loaders