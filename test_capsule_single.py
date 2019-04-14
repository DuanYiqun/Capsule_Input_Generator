import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from pandas import pandas
from torch.autograd import Variable
from torch.optim import Adam
#from torchnet.engine import Engine
#from torchnet.logger import VisdomPlotLogger, VisdomLogger
#from torchvision.utils import make_grid
from tqdm import tqdm
import torchnet as tnt
from Capsule import CapsuleNet_test
import argparse
import torchvision
import torchvision.transforms as transforms
import os
import time

import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
## load model
Dnet = CapsuleNet_test()
Dnet = Dnet.to(device)

print("# parameters:", sum(param.numel() for param in Dnet.parameters()))
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print("==> preparing data ...\n")

transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=12, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=1)


def load_model(modelpath):
    checkpoint = torch.load(modelpath, map_location={'cuda:0': 'cpu'})
    Dnet.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

load_model(modelpath='./train/One-direction-capsule/capsuleone.plk')


npdata = np.zeros((len(testloader),171))

for batch_idx, (inputs, targets) in enumerate(testloader):
    print(batch_idx)
    print('label :{}'.format(targets))
    _,_,outputs, norm = Dnet(inputs)
    if batch_idx <4999:
        npdata[batch_idx*1][:160] = torch.reshape(outputs[0],(1,160)).data.numpy()
        npdata[batch_idx*1][160:170] = torch.reshape(norm[0],(1,10)).data.numpy()
        npdata[batch_idx*1][170:171] = torch.reshape(targets[0],(1,1)).data.numpy()
        npdata[batch_idx+1][:160] = torch.reshape(outputs[0],(1,160)).data.numpy()
        npdata[batch_idx+1][160:170] = torch.reshape(norm[0],(1,10)).data.numpy()
        npdata[batch_idx+1][170:171] = torch.reshape(targets[0],(1,1)).data.numpy()
    #print(torch.reshape(outputs[0],(1,160)))
    #print(norm)

print(npdata)

dataframe = pd.DataFrame(npdata)
dataframe.to_csv('result_dist.csv')