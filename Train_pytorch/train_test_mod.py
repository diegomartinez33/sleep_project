# -*- coding: utf-8 -*-

import time
import argparse
import os.path as osp
import pdb
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

# GPS
import torch.utils.data as data_utils
import numpy as np
import os
import get_metrics as getmet

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# GPS



# from model_nod3 import Net
from Network_FCN import Net_30s
from Network_FCN import Net_10s
from Network_FCN import Net_30s_ST
from Network_FCN import Net_10s_ST

# torch.backends.cudnn.enabled = False

TesisPath = '/hpcfs/home/da.martinez33/Tesis'

ACCpath = osp.join(TesisPath, 'results/trainResults')
modelsPath = osp.join(TesisPath, 'results/trainModels')

dataCreationfolder = osp.join(TesisPath, 'Codes')
sys.path.append(dataCreationfolder)

import createData
import get_metrics as getmet

# --------------------------------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--gamma', type=float, default=0.0005, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')
parser.add_argument('--outf', default=modelsPath, help='folder to output images and model checkpoints')
parser.add_argument('--resume', default='', help="path to model (to continue training)")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# --------------------------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------------------------
ACCpath = osp.join(TesisPath, 'results/trainResults')
modelsPath = osp.join(TesisPath, 'results/trainModels')

data_base_path = osp.join(TesisPath, 'Data', 'DOE_databases', 'balancedData')
data_base_file = 'dataBaseDict_SC_amp_10_0.pkl'

def trainDOEexp(balancedPath=data_base_path,Ei=1):
    """Function to train all combinations for """
    print('Start training\n')
    for file in os.listdir(balancedPath):
        if file.endswith(".pkl"):
            print(file)
            saveComb = file[:-4]  # Folder to save results and models
            for i in range(1,4):
                loadDataBase(balancedPath, file, trainFold=i)

                windowSize = int(file[-8:-6])
                dataType = file[-15:-13]
                if windowSize == 30:
                    if dataType == 'SC':
                        defineModel(Net=Net_30s)
                    else:
                        defineModel(Net=Net_30s_ST)
                elif windowSize == 10:
                    if dataType == 'SC':
                        defineModel(Net=Net_10s)
                    else:
                        defineModel(Net=Net_10s_ST)

                try:
                    for epoch in range(Ei, args.epochs + 1):
                        epoch_start_time = time.time()
                        #testMetrics(saveFolder=saveComb, trainFold=i)
                        train(epoch, saveFolder=saveComb, trainFold=i)
                        test_loss = test(epoch, saveFolder=saveComb, trainFold=i)
                        print('-' * 89)
                        print('| end of epoch {:3d} | time: {:5.2f}s ({:.2f}h)'.format(
                            epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time) / 3600.0))
                        print('-' * 89)
                        saveFolder = osp.join(args.outf, saveComb, str(i))

                        if osp.isdir(saveFolder):
                            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (saveFolder, epoch))  # GPS
                        else:
                            os.makedirs(saveFolder)
                            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (saveFolder, epoch))
                    testMetrics(saveFolder=saveComb, trainFold=i)

                except KeyboardInterrupt:
                    print('-' * 89)
                    print('Exiting from training early')
                    sys.exit()

def testMetrics(saveFolder='',trainFold=1):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.float()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, torch.squeeze(target)).data[0]
        # get the index of the max log-probability
        out_probs = output.data.cpu().numpy()
        pred = output.data.max(1)[1]
        pred2 = pred.cpu().numpy()
        target2 = target.data.cpu().numpy()
        if 'finalPred' in locals():
            finalPred = np.append(finalPred, pred2)

        else:
            finalPred = pred2

        if 'finalTarget' in locals():
            finalTarget = np.append(finalTarget, target2)
        else:
            finalTarget = target2

        if 'finalprobs' in locals():
            finalprobs = np.append(finalprobs, out_probs, axis=0)
        else:
            finalprobs = out_probs


    # ----- Confusion matrix ---------
    confmatPath = osp.join(ACCpath, saveFolder, 'ConfMat_' + str(trainFold) + '.png')
    getmet.get_conf_matrix(finalTarget, finalPred, savePath=confmatPath)
    print('Confusion Matrix created for: \t' + confmatPath)

    # ----- Precision-Recall Curve ---
    prCurvePath_aver = osp.join(ACCpath, saveFolder, 'PR_curve_averaged_' + str(trainFold) + '.png')
    getmet.p_r_curve(finalTarget, finalprobs, averaged='Yes', savePath=prCurvePath_aver)
    prCurvePath_perClass = osp.join(ACCpath, saveFolder, 'PR_curve_classes_' + str(trainFold) + '.png')
    getmet.p_r_curve(finalTarget, finalprobs, averaged='No', savePath=prCurvePath_perClass)
    print('Precision-Recall curve created for: \t' + prCurvePath_aver)
    print('Precision-Recall curve created for: \t' + prCurvePath_perClass)

    # ---- F-score -----
    fscore = getmet.f_score(finalTarget, finalPred, averaged='Yes')
    f_score_path = osp.join(ACCpath, saveFolder, 'F_score_' + str(trainFold) + '.npy')
    f_score_path_text = osp.join(ACCpath, saveFolder, 'F_score_' + str(trainFold) + '.txt')
    np.save(f_score_path, fscore)
    with open(f_score_path_text, 'w') as f:
        f.write(str(fscore))
    print('F-score for: ' + f_score_path + 'is: ' + str(fscore))

    ClassReport = getmet.f_score(finalTarget, finalPred, averaged='No')
    ClassReport_path = osp.join(ACCpath, saveFolder, 'Class_Report_' + str(trainFold) + '.txt')

    with open(ClassReport_path, 'w') as f:
        f.write(ClassReport)
    print('Classification Report: \n\n' + ClassReport)


def loadDataBase(data_path=data_base_path,data_file=data_base_file,trainFold=1):
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

    data_fold1 = Fold1['data']
    label_fold1 = Fold1['Targets']

    data_fold2 = Fold2['data']
    label_fold2 = Fold2['Targets']

    data_fold3 = Fold3['data']
    label_fold3 = Fold3['Targets']

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

    #
    data_train = np.transpose(data_train, (1, 2, 0))
    data_train = np.expand_dims(data_train, axis=1)
    label_train = np.expand_dims(label_train, axis=1)
    label_train = label_train.astype(np.int64)

    data_val = np.transpose(data_val, (1, 2, 0))
    data_val = np.expand_dims(data_val, axis=1)
    label_val = np.expand_dims(label_val, axis=1)
    label_val = label_val.astype(np.int64)

    data_test = np.transpose(data_test, (1, 2, 0))
    data_test = np.expand_dims(data_test, axis=1)
    label_test = np.expand_dims(label_test, axis=1)
    label_test = label_test.astype(np.int64)

    print(data_train.shape)
    print(label_train.shape)
    print(data_val.shape)
    print(label_val.shape)
    print(data_test.shape)
    print(label_test.shape)
    print(data_test.shape[0])
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


def weights_init(m):
    """ custom weights initialization called on netG and netD """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.00, 0.01)
        # m.bias.data.normal_(0.00, 0.1)
        m.bias.data.fill_(0.1)
        # xavier(m.weight.data)
        # xavier(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        # m.weight.data.normal_(1.0, 0.01)
        # m.bias.data.fill_(0)


def defineModel(Net=Net_30s):
    """ Define model net, optimizer and loss criterion """
    global model
    global optimizer
    global criterion
    global load_model
    global res_flag

    model = Net()
    # GPS
    model.apply(weights_init)
    res_flag = 0
    if args.resume != '':  # For training from a previously saved state
        model.load_state_dict(torch.load(args.resume))
        res_flag = 1
    print(model)
    # GPS

    if args.cuda:
        model.cuda()

    load_model = False
    if osp.exists(args.save):
        with open(args.save, 'rb') as fp:
            state = torch.load(fp)
            model.load_state_dict(state)
            load_model = True

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()  # nn.BCELoss().cuda() #nn.SoftMarginLoss()


def train(epoch,saveFolder='',trainFold=1):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.float()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.squeeze(target))
        train_loss += loss.data[0]
        pred = output.data.max(1)[1]
        pred2 = pred.cpu().numpy()
        pred2 = np.expand_dims(pred2, axis=1)
        pred2 = torch.from_numpy(pred2)
        pred2 = pred2.long().cuda()
        pred = pred2
        # ---------------------------------------------
        correct += pred.eq(target.data).cpu().sum()
        acccuracy_batch = 100. * correct / (len(data) * (batch_idx + 1.0))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} ({:.3f})\tAcc: {:.2f}% '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0], train_loss / (batch_idx + 1.0),
                acccuracy_batch))

    train_loss = train_loss
    # loss function already averages over batch size
    train_loss /= len(train_loader)
    acccuracy = 100. * correct / len(train_loader.dataset)
    line_to_save_train = 'Train set: Average loss: {:.4f} Accuracy: {}/{} {:.0f}\n'.format(train_loss,
                                                                                           correct,
                                                                                           len(train_loader.dataset),
                                                                                           acccuracy)

    saveDir = osp.join(osp.join(ACCpath, saveFolder))
    if not osp.isdir(saveDir):
        os.makedirs(saveDir)

    with open(osp.join(ACCpath, saveFolder, 'ACC_train' + str(trainFold) + '.txt'), 'a') as f:
        f.write(line_to_save_train)
    print(line_to_save_train)


def test(epoch,saveFolder='',trainFold=1):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.float()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, torch.squeeze(target)).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        pred2 = pred.cpu().numpy()
        pred2 = np.expand_dims(pred2, axis=1)
        pred2 = torch.from_numpy(pred2)
        pred2 = pred2.long().cuda()
        pred = pred2
        correct += pred.eq(target.data).cpu().sum()
        acccuracy_batch = 100. * correct / (len(data) * (batch_idx + 1.0))
        if batch_idx % args.log_interval == 0:
            print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(val_loader.dataset),
                       100. * batch_idx / len(val_loader), test_loss / (batch_idx + 1.0), acccuracy_batch))
    test_loss = test_loss
    # loss function already averages over batch size
    test_loss /= len(val_loader)
    acccuracy = 100. * correct / len(val_loader.dataset)
    line_to_save_test = 'Test set: Average loss: {:.4f} Accuracy: {}/{} {:.0f}\n'.format(test_loss,
                                                                                         correct,
                                                                                         len(val_loader.dataset),
                                                                                         acccuracy)
    saveDir = osp.join(osp.join(ACCpath, saveFolder))
    if not osp.isdir(saveDir):
        os.makedirs(saveDir)

    with open(osp.join(ACCpath, saveFolder, 'ACC_test' + str(trainFold) + '.txt'), 'a') as f:
        f.write(line_to_save_test)
    print(line_to_save_test)

    return test_loss

def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    best_loss = None
    load_model = False
    res_flag = 0
    if load_model:
        best_loss = test(0)
    # GPS
    if res_flag == 0:
        Ei = 1
    else:
        if args.resume[-6] == '_':
            Ei = int(args.resume[-5]) + 1
            print('-' * 89)
            print('Resuming from epoch %d' % (Ei))
            print('-' * 89)
        else:
            Ei = int(args.resume[-6:-4]) + 1
            print('-' * 89)
            print('Resuming from epoch %d' % (Ei))
            print('-' * 89)
    # GPS
    try:
        trainDOEexp()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        sys.exit()
