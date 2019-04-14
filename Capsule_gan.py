import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable

NUM_ROUTING_ITERATIONS = 3
NUM_CLASSES = 10

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            #GPU
            #logits = Variable(torch.zeros(*priors.size())).cuda() 
            #CPU 
            logits = Variable(torch.zeros(*priors.size()))
            for i in range(self.num_iterations):
                probs = F.softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.confidence = nn.Sequential(
            nn.BatchNorm1d(16 * NUM_CLASSES, 0.8),
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, 0.8),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128, 0.8),
            nn.Linear(128, 1),
            #nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        confidence = self.confidence(x.contiguous().view(x.size(0),-1))
        classes_wtsoft = (x ** 2).sum(dim=-1) ** 0.5 # 将最后一维度 16 加起来 原论文应该是求16 的一届norm 为绝对值之和 这里是二阶 norm
        
        classes = F.softmax(classes_wtsoft, dim=-1) # 原论文并没有softmax
        """
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            max_length_indices = max_length_indices.to('cpu')
            #y = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices.data).cuda()
            y = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices.data)
            #print(y)
            #print(x)
        """
        #reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        
        return classes,classes_wtsoft, x, confidence


class CapsuleNet_separate(nn.Module):
    def __init__(self):
        super(CapsuleNet_separate, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.confidence = nn.Sequential(
            nn.BatchNorm1d(256*20*20, 0.8),
            nn.Linear(256*20*20, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, 0.8),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128, 0.8),
            nn.Linear(128, 1),
            #nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x_0 = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x_0)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        x_0 = x_0.contiguous().view(x_0.size(0),-1)
        confidence = self.confidence(x_0)
        classes_wtsoft = (x ** 2).sum(dim=-1) ** 0.5 # 将最后一维度 16 加起来 原论文应该是求16 的一届norm 为绝对值之和 这里是二阶 norm
        
        classes = F.softmax(classes_wtsoft, dim=-1) # 原论文并没有softmax
        """
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            max_length_indices = max_length_indices.to('cpu')
            #y = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices.data).cuda()
            y = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices.data)
            #print(y)
            #print(x)
        """
        #reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        
        return classes,classes_wtsoft, x, confidence

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 28 // 2**4

        # Output layers
        self.adv_layer = nn.Sequential( nn.Linear(512, 1),
                                        nn.Sigmoid())
        self.aux_layer = nn.Sequential( nn.Linear(512, 10),
                                        nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return label, 1,1,validity 

class Generator(nn.Module):
    def __init__(self, n_classes,latent_dim=16,img_size=28,channels=1):
        super(Generator, self).__init__()

        #self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4# Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim*n_classes, 1568))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 128, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 4, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh()
        )
        """
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            """

    def forward(self, nz):
        gen_input = nz.view(nz.size(0),-1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 32, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Generator_emb(nn.Module):
    def __init__(self,n_classes=10,latent_dim=16,img_size=28,channels=1):
        super(Generator_emb, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4 # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128*self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

if __name__ == "__main__":
    x = torch.randn(2,1,28,28)
    net = CapsuleNet()
    
    predict, logits, capsule,confidence = net(x)
    print(confidence)
    _, max_length_indices = predict.max(dim=1)
    y = Variable(torch.eye(NUM_CLASSES)).index_select( dim=0, index=max_length_indices.data)
    #print(capsule*y[:, :, None])
    adjust = y*predict
    
    net2 = Discriminator()
    predict, logits, capsule,confidence = net2(x)
    print(confidence)
    #print(capsule*y[:, :, None])
    noisevector = torch.randn(2, 10, 16)
    norm2value = torch.norm(noisevector,p=2, dim = -1)
    #print(norm2value)
    scalar = adjust/norm2value
    #print(noise)
    nz = scalar[:,:,None]*noisevector
    #print(torch.norm(nz,p=2, dim = -1))
    print(nz)
    Gnet = Generator(n_classes=10, img_size=28)
    img = Gnet(nz)
    print(img.size())
    
    
    #print(max_length_indices)
    #print(logits)