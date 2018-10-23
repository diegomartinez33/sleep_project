# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# TODO: Cambiar tamaños de kernels y otros
class Net_raw(nn.Module):
    def __init__(self):
        super(Net_raw, self).__init__()

        # first convolutional block
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(480, 9), padding=(0, 4))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        # first pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), padding=0)

        # second convolutional block
        self.conv2 = nn.Conv2d(1, 128, kernel_size=(128, 9), padding=(0, 4))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        # second pooling
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), padding=0)

        # third convolutional block
        self.conv3 = nn.Conv2d(1, 128, kernel_size=(128, 9), padding=(0, 4))
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        # third pooling
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), padding=0)

        # fully convolutional networks
        self.fc1 = nn.Conv2d(1, 100, kernel_size=(128, 10), padding=0)

        self.fc2 = nn.Conv2d(1, 7, kernel_size=(1, 100), padding=0)
        # self.relu5 = nn.ReLU()
        # self.relu6 = nn.ReLU()
        # self.softmax = nn.Softmax2d()
        self.softmax = nn.Softmax()
        # self.fc1 = nn.Linear(128*6*6*6, 128)
        # self.relu13 = nn.ReLU()

    def forward(self, x):
        # first conv block
        out = self.relu1(self.bn1(self.conv1(x)))
        ############### reshape #################
        #out = torch.squeeze(out, 2)
        out = torch.transpose(out, 1, 2)
        #out = torch.unsqueeze(out, 1)  # 16x1x128x640
        #print(out.size())
        out = self.pool1(out)  # 16x1x128x160

        # second block
        out = self.relu2(self.bn2(self.conv2(out)))
        ############### reshape #################
        #out = torch.squeeze(out, 3)
        out = torch.transpose(out, 1, 2)
        #out = torch.unsqueeze(out, 1)  # 16x1x128x160
        #print(out.size())
        out = self.pool2(out)  # 16x1x128x40

        # third conv block
        out = self.relu3(self.bn3(self.conv3(out)))
        #out = torch.squeeze(out, 3)
        out = torch.transpose(out, 1, 2)
        #out = torch.unsqueeze(out, 1)  # 16x1x128x40
        #print(out.size())
        out = self.pool3(out)  # 16x1x128x10

        # fully convolutional layers
        out = self.fc1(out)  # 16x100x1x1 o 16x100xtx1
        out = torch.squeeze(out, 3)
        out = torch.transpose(out, 1, 2)
        out = torch.unsqueeze(out, 1)  # 16x1xtx100
        #print(out.size())

        out = self.fc2(out)  # 16x7xtx1
        out = torch.squeeze(out, 3)
        out = torch.transpose(out, 1, 2)  # 16xtx7
        # out = torch.unsqueeze(out, 1) #16x1xtxn
        # out = torch.squeeze(out, 1) #16xn
        # change softmax input
        out = out.transpose(0, 2)  # nxtxb o 7xtx16
        n, t, b = out.size()
        out = out.contiguous()
        out = out.view(-1, t * b)  # nx16*t
        out = out.t()
        out = self.softmax(out)
        return out
