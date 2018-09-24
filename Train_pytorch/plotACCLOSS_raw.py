#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:08:56 2017

@author: oem
"""

import numpy as np
# import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
    
tesisPath = '/hpcfs/home/da.martinez33/Tesis'
results_path = os.path.join(tesisPath, 'results', 'trainResults_raw')

def makeplot(combPath):
    base_path = os.path.join(results_path, combPath)
    
    epoch = np.linspace(1, 30, num=30)
    
    # Get info from txt files
    train1 = np.genfromtxt(base_path + '/ACC_train1.txt')
    loss1 = train1[:30,4]
    acc1 = train1[:30,7]
    
    test1 = np.genfromtxt(base_path + '/ACC_test1.txt')
    loss_te1 = test1[:30,4]
    acc_te1 = test1[:30,7]
    
    
    train2 = np.genfromtxt(base_path + '/ACC_train2.txt')
    loss2 = train2[:30,4]
    acc2 = train2[:30,7]
    
    test2 = np.genfromtxt(base_path + '/ACC_test2.txt')
    loss_te2 = test2[:30,4]
    acc_te2 = test2[:30,7]
    
    train3 = np.genfromtxt(base_path + '/ACC_train3.txt')
    loss3 = train3[:30,4]
    acc3 = train3[:30,7]
    
    test3 = np.genfromtxt(base_path + '/ACC_test3.txt')
    loss_te3 = test3[:30,4]
    acc_te3 = test3[:30,7]
    
    # ---------------- Plot ACC and Loss info per Fold ----------------------
    fig1=plt.figure()
    plt.plot(epoch, acc1, 'c', label='ACC1_train',color="blue")
    plt.plot(epoch, acc_te1, 'c', label='ACC1_test',color="red")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy for Fold 1')
    plt.legend()
    
    print('Figure 1 ploted')
    
    fig2=plt.figure()
    plt.plot(epoch, loss1, 'c', label = 'Loss1_train',color="blue")
    plt.plot(epoch, loss_te1, 'c', label = 'Loss1_test',color="red")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Fold 1')
    plt.legend()
    
    fig1.savefig(base_path + '/' + 'Fold1_ACC.png')
    fig2.savefig(base_path + '/' + 'Fold1_Loss.png')
    
    print('Figure 2 ploted')
    
    plt.close(fig1)
    plt.close(fig2)
    
    
    fig1=plt.figure()
    plt.plot(epoch, acc2, 'c', label='ACC2_train',color="blue")
    plt.plot(epoch, acc_te2, 'c', label='ACC2_test',color="red")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy for Fold 2')
    plt.legend()
    
    print('Figure 3 ploted')
    
    fig2=plt.figure()
    plt.plot(epoch, loss2, 'c', label = 'Loss2_train',color="blue")
    plt.plot(epoch, loss_te2, 'c', label = 'Loss2_test',color="red")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Fold 2')
    plt.legend()
    
    fig1.savefig(base_path + '/' + 'Fold2_ACC.png')
    fig2.savefig(base_path + '/' + 'Fold2_Loss.png')
    
    print('Figure 4 ploted')
    
    plt.close(fig1)
    plt.close(fig2)
    
    fig1=plt.figure()
    plt.plot(epoch, acc3, 'c', label='ACC3_train',color="blue")
    plt.plot(epoch, acc_te3, 'c', label='ACC3_test',color="red")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy for Fold 3')
    plt.legend()
    
    print('Figure 5 ploted')
    
    fig2=plt.figure()
    plt.plot(epoch, loss3, 'c', label = 'Loss3_train',color="blue")
    plt.plot(epoch, loss_te3, 'c', label = 'Loss3_test',color="red")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Fold 3')
    plt.legend()
    
    fig1.savefig(base_path + '/' + 'Fold3_ACC.png')
    fig2.savefig(base_path + '/' + 'Fold3_Loss.png')
    
    print('Figure 6 ploted')
    
    plt.close(fig1)
    plt.close(fig2)
    
    # ---------------- Plot ACC and Loss info for all folds -------------------
    print('Start plots of all Folds')
    
    fig1 = plt.figure()
    plt.plot(epoch,acc1,color="blue",label='Fold 1')
    plt.plot(epoch,acc2,color="red",label='Fold 2')
    plt.plot(epoch,acc3,color="green",label='Fold 3')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Training Accuracy for Three Folds")
    plt.legend()
    
    print('Figure 1 ploted')
    
    fig2 = plt.figure()
    plt.plot(epoch,acc_te1,color="blue",label='Fold 1')
    plt.plot(epoch,acc_te2,color="red",label='Fold 2')
    plt.plot(epoch,acc_te3,color="green",label='Fold 3')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Validation Accuracy for Three Folds")
    plt.legend()
    
    print('Figure 2 ploted')
    
    fig3 = plt.figure()
    plt.plot(epoch,loss1,color="blue",label='Fold 1')
    plt.plot(epoch,loss2,color="red",label='Fold 2')
    plt.plot(epoch,loss3,color="green",label='Fold 3')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Training Loss for Three Folds")
    plt.legend()
    
    print('Figure 3 ploted')
    
    fig4 = plt.figure()
    plt.plot(epoch,loss_te1,color="blue",label='Fold 1')
    plt.plot(epoch,loss_te2,color="red",label='Fold 2')
    plt.plot(epoch,loss_te3,color="green",label='Fold 3')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Validation Loss for Three Folds")
    plt.legend()
    
    print('Figure 4 ploted')
    
    fig1.savefig(base_path + '/' + 'ACC_Train_3Folds.png')
    fig2.savefig(base_path + '/' + 'ACC_val_3Folds.png')
    fig3.savefig(base_path + '/' + 'Loss_Train_3Folds.png')
    fig4.savefig(base_path + '/' + 'Loss_val_3Folds.png')
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    
    print('plotting of' + combPath + 'Finished')
    
# Create all plots of experiments combinations
# testFolder = 'dataBaseDict_ST_amp_30_3'

# makeplot(testFolder)

# sys.exit()
if os.path.isdir(results_path):
    for folder in os.listdir(results_path):
        folder_path = os.path.join(results_path, folder)
        if os.path.isdir(folder_path):
            makeplot(folder)



