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
from Capsule_gan import CapsuleNet
from Capsule_gan import Generator
from Capsule_gan import Generator_emb
import argparse
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
import os
import time

import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch Single Capsule MNIST Training')
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--mname',default='One-direction-capsule', type=str, help='model name for save')
parser.add_argument('--sample_interval',default=200, type=str, help='interval to sample image')
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

Gnet = Generator_emb()
Gnet = Gnet.to(device)

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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

Gnet.apply(weights_init_normal)
Dnet.apply(weights_init_normal)

## capsule loss
class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        #self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, labels, classes):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        return margin_loss

optimizer_G = torch.optim.Adam(Gnet.parameters(), lr=args.lr, betas=(0.5, 0.99))
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=args.lr, betas=(0.5, 0.99))

## loss definition
adversarial_loss = torch.nn.BCELoss()
capsuleloss = CapsuleLoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

#FloatTensor = torch.cuda.FloatTensor if device=='cuda' else torch.FloatTensor
#LongTensor = torch.cuda.LongTensor if device=='cuda' else torch.LongTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    nz = generate_noise_sample(10,10)
    gen_imgs = Gnet(nz)
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)

def generate_noise_sample(n_row,batches_done):
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    #_, max_length_indices = labels.max(dim=1)
    y = Variable(torch.eye(NUM_CLASSES)).index_select( dim=0, index=labels.data).to(device)
    #print(capsule*y[:, :, None])
    adjust = y*0.8+0.1
    #print(capsule*y[:, :, None])
    noisevector = torch.randn(n_row**2, 10, 16).to(device)
    norm2value = torch.norm(noisevector,p=2, dim = -1).to(device)
    #print(norm2value)
    scalar = adjust/norm2value
    #print(noise)
    nz = scalar[:,:,None]*noisevector
    return nz

#generate_noise_sample(10,10)
#sample_image(10,1)

# ----------
#  Training
# ----------
Dnet.to(device)
Gnet.to(device)

def train(epoch):
    correct = 0
    total = 0
    correct_fake = 0
    total_fake = 0
    for i, (imgs, labels) in enumerate(trainloader):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

         # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        targets = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
        
        labels = labels.to(device)
        targets = targets.to(device)
        real_imgs = real_imgs.to(device)


        # -----------------
        #  Train Generator
        # -----------------
        n_classes = 10
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, 100))))
        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))
        gen_targets = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
        gen_targets.to(device)
        # Generate a batch of images
        gen_imgs = Gnet(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        predict, logits, capsule, confidence = Dnet(gen_imgs)

        g_loss = 0.5 * (adversarial_loss(confidence, valid) + \
                        auxiliary_loss(predict, gen_targets))

        g_loss.backward()
        optimizer_G.step()

        # -----------------
        #  Train Discriminator
        # -----------------

        # Discriminator forward for real images
        optimizer_D.zero_grad()

        # Loss for real images
        predict, logits, capsule,confidence = Dnet(real_imgs)
        d_real_loss =  (adversarial_loss(confidence, valid) + auxiliary_loss(predict , targets)) / 2
        
        # Loss for fake images
        predict_fake, logits_fake, capsule_fake,confidence_fake = Dnet(gen_imgs.detach())
        Dnet(gen_imgs.detach())
        d_fake_loss =  (adversarial_loss(confidence_fake, fake) + \
                        auxiliary_loss(predict_fake, gen_targets)) / 2

        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        # random sample generators
        _, max_length_indices = predict.max(dim=1)

        total += targets.size(0)
        correct += max_length_indices.eq(labels).sum().item()

        _,predicted_labels_fake = predict_fake.max(dim=1)
        total_fake += predict.size(0)
        correct_fake += predicted_labels_fake.eq(max_length_indices).sum().item()

        real_acc = 100.*correct/total
        fake_acc = 100.*correct_fake/total_fake 
        
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, real_acc: %d%%, fake_acc:%d%%] [G loss: %f]" % (epoch, NUM_EPOCHS, i, len(trainloader),
                                                            d_loss.item(), real_acc, fake_acc,
                                                            g_loss.item()))
        batches_done = epoch * len(trainloader) + i
        if batches_done % args.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
        
    state = {
        'Dnet': Dnet.state_dict(),
        'Gnet': Gnet.state_dict(),
        'real_acc': real_acc,
        'epoch': epoch
    }
    
    torch.save(state,'capsule_gan_check.plk')
    
        

for epoch in range(start_epoch, start_epoch+60):
    train(epoch)





        


