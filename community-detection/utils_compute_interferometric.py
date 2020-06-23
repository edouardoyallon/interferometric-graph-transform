
import torch


from utils import *
import torch.optim as optim
from visdom import Visdom
import argparse
import collections


batch=64

def loss(x,z):
    norm_x =  torch.sum(x.view(x.size(0),-1)**2,1)
    norm_net_x =  torch.sum(z.view(z.size(0),-1)**2,1)
    to_reg= torch.mean(norm_x)-torch.mean(norm_net_x)
    easy = torch.mean(torch.sqrt(norm_net_x)/(1e-4+torch.sqrt(norm_x)))
    return torch.mean(to_reg), easy


def compute_representation(adj,X,K=5,epochs=5):
    o=0

    adj, F, Fc, Vect0, averaging = Fourier_atoms(adj, 1)
    net = []
    for i in range(o + 2):
        net.append(InterferometricModuleGraphs(averaging, F, X.shape[2], K=K, size_avg=1))

    for epoch in range(epochs):
        indices = collections.deque()
        Z = np.arange(X.shape[0])

        np.random.shuffle(Z)
        indices.extend(Z)
        n=0
        # print(epoch)# loop over the dataset multiple times
        while len(indices) >= batch:
            batch_idx = [indices.popleft() for i in range(batch)]

            inputs =  torch.from_numpy(X[batch_idx, :])

            if n==0:
                optimizer = optim.SGD(net[o].parameters(), lr=0.1, momentum=0.0)

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

    return net


def compute_feature(net, X, no_averaging=False):
    o=0

    outs = []
    HF = torch.from_numpy(X)
    for p in range(o + 2):
        if p == o + 1:
            out = net[p](HF, only_averaging=True, no_averaging=no_averaging)
        else:
            HF, out = net[p](HF,no_averaging=no_averaging)
        outs.append(out)

    output = torch.cat(outs, 1)
    return output.detach().view(output.size(0),-1)

