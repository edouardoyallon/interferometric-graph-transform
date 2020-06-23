import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import gzip
import pickle

import numpy as np
from utils import *
import torch.optim as optim
from visdom import Visdom
import argparse

viz = Visdom(port=8050)

torch.manual_seed(0)
import numpy as np
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch=64

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

parser = argparse.ArgumentParser(description='Images')
parser.add_argument('--K', type=int, default=1,help='scattering 1st or 2nd order')
parser.add_argument('--J', type=int, default=1, help='scattering 1st or 2nd order')
parser.add_argument('--dataset', type=str, default='cifar',help='dataset type')
parser.add_argument('--lr_schedule', type=str, default='{0:1e4, 500:1e3, 1000:1e2,1500:10,2000:1,2500:0.1}')
parser.add_argument('--epochs', type=int, default=5)
args = parser.parse_args()

import ast
learning_rates = ast.literal_eval(args.lr_schedule)

N=0
if args.dataset == 'cifar':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)
    N=32
elif args.dataset == 'mnist':
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data_mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data_mnist', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch, shuffle=False)
    N=28


net = InterferometricModule(K=args.K,J=args.J,N=N)#,avg=args.avg)
if torch.cuda.is_available():
    net=net.cuda()
net.projection_l2_ball()


# This loss compute \sum_x ||x||^ 2 - ||A|Wx| ||^2 and then ||A|Wx|||/||x|| for visualization purposes
def loss(x,z):
    norm_x =  torch.sum(x.view(x.size(0),-1)**2,1)
    norm_net_x =  torch.sum(z.view(z.size(0),-1)**2,1)
    to_reg= torch.mean(norm_x)-torch.mean(norm_net_x)
    easy = torch.mean(torch.sqrt(norm_net_x)/(1e-4+torch.sqrt(norm_x)))
    return torch.mean(to_reg), easy



loss_window = viz.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='Loss',
                                 title='Training Loss',
                                 legend=['Loss']))

loss_window_2 = viz.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='Loss',
                                 title='readable Loss',
                                 legend=['Loss']))

loss_norm = viz.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='minibatches',
                                 ylabel='Loss',
                                 title='norm',
                                 legend=['Loss']))


image = viz.image(torch.zeros(3,N,N))
image2 = viz.image(torch.zeros(3,150,150))
n=0

# The LR must start pretty high!

for epoch in range(args.epochs):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # This must be adapted

        if n in learning_rates:
            optimizer = optim.SGD(net.parameters(), lr=learning_rates[n], momentum=0.0)
            print(str(n)+':'+str(learning_rates[n]))



        # get the inputs
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()

        output = net(inputs)
        loss_isometry, easy = loss(inputs, output)
        loss_isometry.backward()
        optimizer.step()
        a=net.projection_l2_ball()



        # we now displzy all the available data
        loss_isometry,easy = loss(inputs,output)

        viz.line(
            X=torch.ones((1, 1)).cpu() * n,
            Y=torch.Tensor([loss_isometry]).unsqueeze(0),
            win=loss_window,
            update='append')
        viz.line(
            X=torch.ones((1, 1)).cpu() * n,
            Y=torch.Tensor([a]).unsqueeze(0),
            win=loss_norm,
            update='append')
        viz.line(
            X=torch.ones((1, 1)).cpu() * n,
            Y=torch.Tensor([easy]).unsqueeze(0),
            win=loss_window_2,
            update='append')
        n=n+1


        glah = torch.sum(net.W_real**2+net.W_imag**2,0).sqrt().view(N,N).cpu()
        glah2= torch.sqrt(net.W_real**2+net.W_imag**2).cpu()
        glah2 = torch.from_numpy(np.fft.fftshift(glah2.detach().numpy(),axes=(1,2)))
        glah2=torchvision.utils.make_grid(glah2.unsqueeze(1))
        glah2 = glah2/torch.max(glah2)
        glah = torch.from_numpy(np.fft.fftshift(glah.detach().numpy(),axes=(0,1)))
        glah = glah/torch.max(glah)

        viz.image(glah, win=image)
        viz.image(glah2, win=image2)

state={}
state.update({'W_real':net.W_real,'W_imag':net.W_imag})
torch.save(state, 'W_'+args.dataset+'_'+args.lr_schedule+'_'+str(args.J)+'_'+str(args.K)+'.t7')

print('Training done')
