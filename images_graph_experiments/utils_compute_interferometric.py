
import torch


from utils import *
import torch.optim as optim
from visdom import Visdom
import argparse
import collections
viz = Visdom(port=8050)
from utils_compute_interferometric import *
batch=64

def loss(x,z):
    norm_x =  torch.sum(x.view(x.size(0),-1)**2,1)
    norm_net_x =  torch.sum(z.view(z.size(0),-1)**2,1)
    to_reg= torch.mean(norm_x)-torch.mean(norm_net_x)
    easy = torch.mean(torch.sqrt(norm_net_x)/(1e-4+torch.sqrt(norm_x)))
    return torch.mean(to_reg), easy


def compute_representation(adj,X,K=30):
    o=0

    adj, F, Fc, Vect0, averaging = Fourier_atoms(adj, 1)
    net = []
    for i in range(o + 1):
        net.append(InterferometricModuleGraphs(averaging, F, X.shape[-2], K=K, size_avg=1))

    for epoch in range(5):
        indices = collections.deque()
        Z = np.arange(X.shape[0])

        np.random.shuffle(Z)
        indices.extend(Z)
        n=0
        while len(indices) >= batch:
            batch_idx = [indices.popleft() for i in range(batch)]
            inputs =  X[batch_idx, :]
            if n==0:
                optimizer = optim.SGD(net[o].parameters(), lr=0.1, momentum=0.0)
            inputs = inputs.permute((0, 2, 1)).contiguous()

            optimizer.zero_grad()
            outs = []
            HF = inputs
            for p in range(o + 2):
                if p == o + 1:
                    out = net[p](HF, only_averaging=True)
                else:
                    HF, out = net[p](HF)
                outs.append(out)

            output = torch.cat(outs, 1)

            loss_isometry, easy = loss(inputs, output)

            loss_isometry.backward()
            optimizer.step()
            a = net[o].projection_l2_ball()
    print('Training done')
    return net


def compute_feature(X, net):
    o=0

    outs = []
    HF = X
    for p in range(o + 2):
        if p == o + 1:
            out = net[p](HF, only_averaging=True)
        else:
            HF, out = net[p](HF)
        outs.append(out)

    output = torch.cat(outs, 1)
    print('Features computed')
    return output

