
import torch
import graphs

from utils import *
import torch.optim as optim
from visdom import Visdom
import argparse
import collections
viz = Visdom(port=8050)

batch=64


parser = argparse.ArgumentParser(description='Graphs skeletons')
parser.add_argument('--K', type=int, default=1,help='number of features')
parser.add_argument('--order', type=int, default=1,help='interferometric order')
parser.add_argument('--lr_schedule', type=str, default='{0:1e4, 500:1e3, 1000:1e2,1500:10,2000:1,2500:0.1}')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--data_name', type=str, help='dataset : SBU, NTU', default='SBU')
parser.add_argument('--split', type=int, default=1, help='SBU : 1,2,3, 4, 5')
args = parser.parse_args()

import ast
learning_rates = ast.literal_eval(args.lr_schedule)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(0)
import numpy as np
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset = graphs.Data()

if args.data_name == 'SBU':
    dataset = dataset.get_SBU(args)
    N_nodes = dataset[-1].shape[0]
    size_avg = 2

if args.data_name == 'NTU_xview_values':
    dataset = dataset.get_NTU_xview_values()
    N_nodes = dataset[-1].shape[0]
    size_avg = 2
if args.data_name == 'NTU_xsub_values':
    dataset = dataset.get_NTU_xview_values(view='xsub')
    N_nodes = dataset[-1].shape[0]
    size_avg = 2

# get dataset
labels_train, features_train, labels_test, features_test, adj = dataset

adj,F,Fc,Vect0,averaging=Fourier_atoms(adj,size_avg)
print(F.shape)
net=[]

for i in range(args.order+1):
    net.append(InterferometricModuleGraphs(averaging, F, features_train.shape[-2],K=args.K,size_avg=size_avg))
    if torch.cuda.is_available():
        net[i] = net[i].cuda()
        net[i].avg=net[i].avg.cuda()
        net[i].F_r= net[i].F_r.cuda()
        net[i].F_i =  net[i].F_i.cuda()
        net[i].F_inv_r =  net[i].F_inv_r.cuda()
        net[i].F_inv_i =  net[i].F_inv_i.cuda()

# This loss compute \sum_x ||x||^ 2 - ||A|Wx| ||^2 and then ||A|Wx|||/||x|| for visualization purposes
def loss(x,z):
    norm_x =  torch.sum(x.view(x.size(0),-1)**2,1)
    norm_net_x =  torch.sum(z.view(z.size(0),-1)**2,1)
    to_reg= torch.mean(norm_x)-torch.mean(norm_net_x)
    easy = torch.mean(torch.sqrt(norm_net_x)/(1e-4+torch.sqrt(norm_x)))
    return torch.mean(to_reg), easy
# independant counter
n=0

# The LR must start pretty high!


np.random.seed(0)
indices = collections.deque()
indices.extend(np.arange(features_train.shape[0]))

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


for o in range(args.order):
    N=n

    for epoch in range(args.epochs):
        indices = collections.deque()
        Z = np.arange(features_train.shape[0])

        np.random.shuffle(Z)
        indices.extend(Z)
        # print(epoch)# loop over the dataset multiple times
        while len(indices) >= batch:
            batch_idx = [indices.popleft() for i in range(batch)]

            train_y, train_x = labels_train[batch_idx], features_train[batch_idx, :]
            train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
            train_y = torch.from_numpy(train_y).type(torch.LongTensor)

            inputs, target = train_x.to(device), train_y.to(device)

            if n-N in learning_rates:
                optimizer = optim.SGD(net[o].parameters(), lr=learning_rates[n-N], momentum=0.0)
                print(str(n) + ':' + str(learning_rates[n-N]))

            inputs = inputs.permute((0, 2, 1)).contiguous()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            optimizer.zero_grad()
            outs = []
            HF = inputs

            for p in range(o+2):
                if p==o+1:
                    out = net[p](HF,only_averaging=True)
                else:
                    HF, out = net[p](HF)
                outs.append(out)

            output = torch.cat(outs,1)
            loss_isometry, easy = loss(inputs, output)
            loss_isometry.backward()
            optimizer.step()
            a = net[o].projection_l2_ball()

            # we now displzy all the available data
            loss_isometry, easy = loss(inputs, output)

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
            n = n + 1

            glah = torch.sum(net[o].W_real ** 2 + net[o].W_imag ** 2, 0).sqrt().cpu()

state={}
state.update({'averaging':averaging,'F':F})
for i in range(args.order):
    state.update({'W_'+str(i)+'real':net[i].W_real,'W_'+str(i)+'imag':net[i].W_imag,})

name = './W_' + args.lr_schedule + '_' + str(args.K) + '_' + args.data_name + '_' + str(args.split) + '_'+\
       str(args.order)+'_'+str(args.epochs)+'.t7'
print(name)

torch.save(state, name)
print('Finished Training')
