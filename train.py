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
from CapsuleGan import CapsuleNet
import argparse
import torchvision
import torchvision.transforms as transforms
import os
import time

import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch Single Capsule MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--mname',default='One-direction-capsule', type=str, help='model name for save')
args = parser.parse_args()

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3

## load model
Dnet = CapsuleNet()
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
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('./train/'+args.mname+'/checkpoints'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./train/'+args.mname+'/checkpoints/best_check.plk')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(best_acc)

trainpd = pd.DataFrame({"epoch":"","accuracy":"","loss":""},index=["0"])
savepath='./train/'+str(args.mname)+'/checkpoints/'

if not os.path.isdir(savepath):
    os.makedirs(savepath)

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)

optimizer = Adam(Dnet.parameters(),lr = 0.01, weight_decay=5e-4)

capsuleloss = CapsuleLoss()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    Dnet.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time=time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = torch.eye(NUM_CLASSES).index_select(dim=0, index=targets)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, reconstructions = Dnet(inputs, targets)
        loss = capsuleloss(inputs,targets,outputs,reconstructions)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accuracy=100.*correct/total
#       progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), accuracy, correct, total))
    end_time=time.time()
    epoch_time=end_time-start_time
    data=[epoch,accuracy,train_loss/(batch_idx+1),epoch_time]
    print('trainloss:{},accuracy:{},time_used:{}'.format(train_loss/(batch_idx+1),accuracy,epoch_time))
    state = {
            'net': Dnet.state_dict(),
            'acc': accuracy,
            'epoch': epoch
        }
    """
    if epoch % 30 == 0:
        savepath='./train/'+str(args.mname)+'/checkpoints/'+str(epoch)+'_check.plk'
        print('system_saving...at {} epoch'.format(epoch))
        torch.save(state, savepath)
    """
    return data
  #  new = pd.DataFrame({"epoch":epoch,"accuracy":accuracy,"loss":train_loss/(batch_idx+1)},index=["0"])
    # trainpd = trainpd.append(new,newignore_index=True)

def test(epoch):
    global best_acc
    Dnet.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time=time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, reconstructions = Dnet(inputs, targets)
        loss = capsuleloss(inputs,targets,outputs,reconstructions)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accuracy=100.*correct/total

    end_time=time.time()
    epoch_time=end_time-start_time   
#        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    data=[epoch,accuracy,test_loss/(batch_idx+1),epoch_time]
    print('testloss:{},accuracy:{},time_used:{}'.format(test_loss/(batch_idx+1),accuracy,epoch_time))
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..best_record')
        state = {
            'net': Dnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        #if not os.path.isdir('checkpoint'):
            #os.mkdir('checkpoint')
        torch.save(state, savepath)
        best_acc = acc
    return data


a=[1,2,3,4]
trainnp=np.array(a)
testnp=np.array(a)


for epoch in range(start_epoch, start_epoch+90):
    nd = train(epoch)
    ed = test(epoch)
    trainnp=np.vstack((trainnp,np.array(nd)))
    testnp=np.vstack((testnp,np.array(ed)))

savepath='../outputs/'+str(args.mname)+'train01.csv'
train_data=pd.DataFrame(trainnp,columns=['epoch','accuracy','loss','epoch_time'])
train_data.to_csv(savepath)
savepath='../outputs/'+str(args.mname)+'test01.csv'
test_data=pd.DataFrame(testnp,columns=['epoch','accuracy','loss','epoch_time'])
test_data.to_csv(savepath)
