# -*- coding: utf-8 -*-
"""ResNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sP0LhM2JPoibZ3vXlNLsguJh6dbpF3xk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available')
else:
    print('CUDA is available')
     
path = '/model_file'

# set the hyperparameter
batch_size = 64
learning_rate = 0.01
epoch_num = 100


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def project1_model():
    return ResNet(BasicBlock, [2, 2, 2, 2])

net = project1_model().cuda()
loss = nn.CrossEntropyLoss()

train_loss_history = np.zeros([20, 20, 20, 100])
test_loss_history = np.zeros([20, 20, 20, 100])
train_acc_history = np.zeros([20, 20, 20, 100])
test_acc_history = np.zeros([20, 20, 20, 100])

test_acc = np.zeros([20, 20, 20])

from sklearn.model_selection import KFold
k_fold = 5

# hyperparameters array
batch_sizes = 512
learning_rates = np.logspace(-5, -1, num=20, endpoint=False)
epoch_num = 1
momentums = np.linspace(0.5, 1, num=20, endpoint=False)
weight_decays = np.logspace(-5, -1, num=20, endpoint=True)[::-1]

kfold = KFold(n_splits=k_fold, shuffle=True)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))])
data = datasets.CIFAR10('data', download=True, transform=transform)
for rate_id, rate in enumerate(learning_rates):
    for mom_id, mom in enumerate(momentums):
        for weight_id, weight in enumerate(weight_decays):
            # Start print
            print('-------------------------------------')

            optimizer = torch.optim.SGD(net.parameters(), lr=rate, momentum=mom, weight_decay=weight)
            #kfold Cross validation

            accs = []
            for fold, (ind_train, ind_test) in enumerate(kfold.split(data)):
              print(f'FOLD {fold}');
              print('-------------------------------------')
              train_subsampler = torch.utils.data.SubsetRandomSampler(ind_train)
              test_subsampler = torch.utils.data.SubsetRandomSampler(ind_test)

              train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_subsampler)
              # vali_loader = torch.utils.data.DataLoader(validata, batch_size=batch_size, shuffle=True)
              test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=test_subsampler)
              max_acc = 0
              

              for epoch in range(epoch_num):
              # print(epoch)
                  train_loss = 0.0
                  train_acc = 0.0
                  net.train()
                  for i, train_data in enumerate(train_loader):
                      inputs, labels = train_data
                      inputs, labels = inputs.cuda(), labels.cuda()
                      optimizer.zero_grad()
                      
                      # forward + backward
                      predicted_output = net(inputs)
                      fit = loss(predicted_output, labels)
                      fit.backward()
                      optimizer.step()
                      train_loss += fit.item()
                      _, pred = predicted_output.max(1)
                      num_correct = (pred==labels).sum().item()
                      acc = num_correct / inputs.shape[0]
                      train_acc += acc

                  train_loss = train_loss/len(train_loader)
                  train_loss_history[rate_id, mom_id, weight_id, epoch] = train_loss
                  train_acc = train_acc / len(train_loader)
                  train_acc_history[rate_id, mom_id, weight_id, epoch] = train_acc
                  
                  test_loss = 0.0
                  test_acc = 0.0
                  net.eval()
                  for j, data in enumerate(test_loader):
                      with torch.no_grad():
                          images, labels = data
                          images = images.cuda()
                          labels = labels.cuda()
                          predicted_output = net(images)
                          fit = loss(predicted_output,labels)
                          test_loss += fit.item()
                          _, pred = predicted_output.max(1)
                          num_correct = (pred==labels).sum().item()
                          acc = num_correct / images.shape[0]
                          test_acc += acc
                  test_loss = test_loss / len(test_loader)
                  test_loss_history[rate_id, mom_id, weight_id, epoch] = test_loss
                  test_acc = test_acc / len(test_loader)
                  test_acc_history[rate_id, mom_id, weight_id, epoch] = test_acc
                  max_acc = max(test_acc, max_acc)
                  accs.append(max_acc)
                  print('Epoch %s, Train loss %.6f, Test loss %.6f, Train acc %.6f, Test acc %.6f'%(epoch, train_loss, test_loss, train_acc*100, test_acc*100))

            average_acc = np.mean(accs)
            test_acc[rate_id, mom_id, weight_id] = average_acc
# torch.save({'model':net.state_dict()}, './model_file/project1_model.pt')