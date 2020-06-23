import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
# from kymatio import Scattering2D
# import kymatio.datasets as scattering_datasets
import torch
import argparse
import torch.nn as nn
from utils import *
import graphs
from numpy import linalg as LA
import time
import collections
import networkx as nx
import numpy as np
import os
import ast
from utils_classif import *
import matplotlib
import numpy.linalg as linalg

matplotlib.use('agg')
import matplotlib.pyplot as plt

seed = 42
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import numpy as np
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(seed)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(seed)


class myarray(np.ndarray):
    @property
    def H(self):
        return self.conj().T

def train_fc(model, linear,BN, device, train_loader, optimizer, epoch, order,C,avg):
    BN.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            outs = []
            HF = data
            for p in range(order + 1):
                if p == order + 1:
                    out = model[p](HF, only_averaging=True,no_averaging=avg)
                else:
                    HF, out = model[p](HF,no_averaging=avg)

                outs.append(out)

        output = torch.cat(outs, 1).detach().view((data.size(0), -1))
        output = BN(output)
        output = linear(output)
        y = F.one_hot(target,num_classes=linear.weight.data.shape[0])-0.5
        loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
        loss += C * torch.mean(linear.weight ** 2)  # l2 penalty
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_fc(model, linear, BN,device, test_loader, epoch, order,avg):
    test_loss = 0
    correct = 0
    BN.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outs = []
            HF = data
            for p in range(order + 1):
                if p == order + 1:
                    out = model[p](HF, only_averaging=True,no_averaging=avg)
                else:
                    HF, out = model[p](HF,no_averaging=avg)

                outs.append(out)

            output = torch.cat(outs, 1).detach().view((data.size(0), -1))
            output = BN(output)
            output = linear(output)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))




def Linear_SVM_classification():
    parser = argparse.ArgumentParser(description='Graph skeletons classification with Linear SVM')
    parser.add_argument('--mode', type=str, default='svm', help='svm or fc')
    parser.add_argument('--order', type=int, default=1, help='order to use')
    parser.add_argument('--C', type=float, default=1e-2, help='regularization svm')
    parser.add_argument('--data_name', type=str, help='dataset : SBU, NTU', default='SBU')
    parser.add_argument('--lr_schedule', type=str, default='{0:0.01,25:0.001,50:0.0001}', help='lr schedule for FC')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for FC')
    parser.add_argument('--split', type=int, default=1, help='SBU : 1,2,3, 4, 5')
    parser.add_argument('--file', type=str, help = 'file to load with the parameters of the model')
    parser.add_argument('--K', type=int, help = 'number of features used')
    parser.add_argument('--no_averaging', action='store_true')

    args = parser.parse_args()

    K = args.K

    dataset = graphs.Data()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.data_name == 'SBU':

        dataset = dataset.get_SBU(args)
        N_nodes = dataset[-1].shape[0]
        N_inputs = 30
        size_avg = 2
        n_classes = 21

    if args.data_name == 'NTU_xview_values':
        dataset = dataset.get_NTU_xview_values()
        N_nodes = dataset[-1].shape[0]
        N_inputs = 50
        size_avg = 2
        n_classes=5

    if args.data_name == 'NTU_xsub_values':
        dataset = dataset.get_NTU_xview_values(view='xsub')
        N_nodes = dataset[-1].shape[0]
        size_avg = 2


    print('Linear classifier:')
    # get dataset
    labels_train, features_train, labels_test, features_test, _ = dataset

    # Load learned filters W_real and W_imag
    state = torch.load(args.file)


    averaging, F = state['averaging'], state['F']
    device='cpu'
    net = []

    for i in range(args.order +1):
        net.append(InterferometricModuleGraphs(averaging, F, N_inputs, K=args.K, size_avg=size_avg))
        if i<args.order:
            net[i].W_real.data = state['W_'+str(i)+'real'].cpu()
            net[i].W_imag.data = state['W_' + str(i) + 'imag'].cpu()
        if torch.cuda.is_available(): # and not args.mode=='svm':
            device='cuda'
            net[i] = net[i].cuda()
            net[i].avg=net[i].avg.cuda()
            net[i].F_r= net[i].F_r.cuda()
            net[i].F_i =  net[i].F_i.cuda()
            net[i].F_inv_r =  net[i].F_inv_r.cuda()
            net[i].F_inv_i =  net[i].F_inv_i.cuda()

    features_train = torch.from_numpy(features_train).type(torch.FloatTensor)
    features_test = torch.from_numpy(features_test).type(torch.FloatTensor)

    outs = []
    HF = features_train[0,...].permute([1,0]).unsqueeze(0).to(device)

    print(HF.is_cuda)
    # we determine the size of the otuput features
    for p in range(args.order + 1):
        if p == args.order + 1:
            out = net[p](HF, only_averaging=True,no_averaging=args.no_averaging)
        else:
            HF, out = net[p](HF,no_averaging=args.no_averaging)
        outs.append(out)
    SIZE_feature = torch.cat(outs, 1).detach().numel()


    features_train = features_train.permute((0, 2, 1)).contiguous()
    features_test = features_test.permute((0, 2, 1)).contiguous()
    if args.mode == 'svm':
        train_loader = torch.utils.data.TensorDataset(features_train, torch.from_numpy(labels_train))
        train_loader = torch.utils.data.DataLoader(train_loader, batch_size=64, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.TensorDataset(features_test, torch.from_numpy(labels_test))
        test_loader = torch.utils.data.DataLoader(test_loader, batch_size=64, shuffle=False, num_workers=2)

        N_sample_train = len(train_loader.dataset)
        N_sample_test = len(test_loader.dataset)

        X_train = torch.zeros(N_sample_train, SIZE_feature).float()
        y_train = torch.zeros(N_sample_train).float()
        X_test = torch.zeros(N_sample_test, SIZE_feature).float()
        y_test = torch.zeros(N_sample_test).float()
        b = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            HF = data
            outs = []
            with torch.no_grad():
                for p in range(args.order + 1):
                    if p == args.order + 1:
                        out = net[p](HF, only_averaging=True, no_averaging=args.no_averaging)
                    else:
                        HF, out = net[p](HF, no_averaging=args.no_averaging)
                    outs.append(out)
                output = torch.cat(outs, 1).detach().view((data.size(0), -1))
            X_train[b:b + output.size(0), :] = output.view(output.size(0),-1).cpu()
            y_train[b:b + output.size(0)] = target.cpu()
            b += output.size(0)

        b = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            HF = data
            outs = []
            with torch.no_grad():
                for p in range(args.order + 1):
                    if p == args.order + 1:
                        out = net[p](HF, only_averaging=True, no_averaging=args.no_averaging)
                    else:
                        HF, out = net[p](HF, no_averaging=args.no_averaging)
                    outs.append(out)

                output = torch.cat(outs, 1).detach().view((data.size(0), -1))

            X_test[b:b + output.size(0), :] = output.view(output.size(0),-1).cpu()
            y_test[b:b + output.size(0)] = target.cpu()
            b += output.size(0)

        accValid_with_interferometric = SVM(y_train.numpy(), X_train.numpy(), y_test.numpy(),X_test.numpy(),args.C)
    elif args.mode == 'fc':



        learning_rates = ast.literal_eval(args.lr_schedule)
        train_loader = torch.utils.data.TensorDataset(features_train,torch.from_numpy(labels_train))
        train_loader = torch.utils.data.DataLoader(train_loader, batch_size=256, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.TensorDataset(features_test, torch.from_numpy(labels_test))
        test_loader = torch.utils.data.DataLoader(test_loader, batch_size=256, shuffle=False, num_workers=2)
        BN = nn.BatchNorm1d(SIZE_feature,affine=False).to(device)
        linear = nn.Linear( SIZE_feature, n_classes).to(device)
        
        optimizer = None

        for epoch in range(args.epochs):
            if epoch in learning_rates:
                optimizer = optim.Adam(linear.parameters(), lr=learning_rates[epoch])
                print(str(epoch) + ':' + str(learning_rates[epoch]))
            train_fc(net, linear,BN, device, train_loader, optimizer, epoch, args.order, args.C, args.no_averaging)
            test_fc(net, linear,BN, device, test_loader, epoch, args.order,args.no_averaging)




    print('accuracy on : ', str(args.data_name), '_split', str(args.split), 'with interferometric is :',
          str(accValid_with_interferometric))



if __name__ == '__main__':

    Linear_SVM_classification()
