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
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(1, 9), padding=(0, 4))
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        # second pooling
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), padding=0)

        # third convolutional block
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(1, 9), padding=(0, 4))
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        # third pooling
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4), padding=0)

        # fully convolutional networks
        self.fc1 = nn.Conv2d(128, 100, kernel_size=(1, 10), padding=0)

        self.fc2 = nn.Conv2d(100, 7, kernel_size=(1, 1), padding=0)
        # self.relu5 = nn.ReLU()
        # self.relu6 = nn.ReLU()
        # self.softmax = nn.Softmax2d()
        self.softmax = nn.Softmax()
        # self.fc1 = nn.Linear(128*6*6*6, 128)
        # self.relu13 = nn.ReLU()

    def forward(self, x):
        # first conv block
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.pool1(out)  # 16x128x120x160
        #print(out.size())
        #time.sleep(5)

        # second block
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.pool2(out)  # 16x128x30x40

        # third conv block
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.pool3(out)  # 16x128x7x10
        #print(out.size())

        # fully convolutional layers
        out = self.fc1(out)  # 16x100x1x1
        out = self.fc2(out)  # 16x7x1x1
        out = torch.squeeze(out, 3)
        out = torch.transpose(out, 1, 2)  # 16x1x7
        out = out.transpose(0, 2)  # 7x1x16
        n, t, b = out.size()
        out = out.contiguous()
        #print(out.size())
        #time.sleep(5)
        # out = out.view(out.size(0), -1)
        out = out.view(-1, t * b)  # 7x16*t
        out = out.t()
        out = self.softmax(out)

        return out
