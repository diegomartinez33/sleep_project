  #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np

#read .mat files
train_scaled_dicc = sio.loadmat('/media/user_home2/EEG/Sleep/30s_database/TrainImages/Train_cated', appendmat = True)
train_data = train_scaled_dicc['AllTrain']

train_data = train_data.astype(np.int64)

train_label_dicc = sio.loadmat('/media/user_home2/EEG/Sleep/30s_database/TrainImages/Train_labels', appendmat = True)
train_label =  train_label_dicc['TrainLabels']


val_scaled_dicc = sio.loadmat('/media/user_home2/EEG/Sleep/30s_database/ValImages/Val_cated', appendmat = True)
val_data = val_scaled_dicc['AllVal']

val_data = val_data.astype(np.int64)

val_label_dicc = sio.loadmat('/media/user_home2/EEG/Sleep/30s_database/ValImages/Val_labels', appendmat = True)
val_label =  val_label_dicc['ValLabels']


test_scaled_dicc = sio.loadmat('/media/user_home2/EEG/Sleep/30s_database/TestImages/Test_cated', appendmat = True)
test_data = test_scaled_dicc['AllTest']

test_data = test_data.astype(np.int64)

test_label_dicc = sio.loadmat('/media/user_home2/EEG/Sleep/30s_database/TestImages/Test_labels', appendmat = True)
test_label =  test_label_dicc['TestLabels']

#save as .npy files
np.save('/media/user_home2/EEG/Sleep/30s_database/train_data.npy', train_data)
np.save('/media/user_home2/EEG/Sleep/30s_database/train_label.npy', train_label)

np.save('/media/user_home2/EEG/Sleep/30s_database/val_data.npy', val_data)
np.save('/media/user_home2/EEG/Sleep/30s_database/val_label.npy', val_label)

np.save('/media/user_home2/EEG/Sleep/30s_database/test_data.npy', test_data)
np.save('/media/user_home2/EEG/Sleep/30s_database/test_label.npy', test_label)