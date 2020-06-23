import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from kymatio import Scattering2D
import kymatio.datasets as scattering_datasets
import torch
import argparse
import ast
import torch.nn as nn
from utils import *

from utils_classif import *

torch.manual_seed(0)
import numpy as np
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






def train(model, device, train_loader, optimizer, epoch, feature,C,bn=None):
    model.train()
    bn.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            output = feature(data)

        output = output.view(output.size(0),-1)
        output=bn(output)
        output = model(output)
        y = F.one_hot(target, num_classes=model.weight.data.shape[0]) - 0.5
        loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
        loss += C * torch.mean(model.weight ** 2)  # l2 penalty
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, feature,bn=None):
    model.eval()
    bn.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = feature(data)
            output = output.view(output.size(0),-1)
            output=bn(output)
            output = model(output)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description='Images classification')
    parser.add_argument('--mode', type=int, default=1,help='scattering 1st or 2nd order')

    parser.add_argument('--J', type=int, default=1, help='scattering 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='fc',help='classifier model')
    parser.add_argument('--dataset', type=str, default='cifar', help='dataset type')
    parser.add_argument('--feature', type=str, default='scattering', help='classifier model')
    parser.add_argument('--path', type=str, default='scattering', help='classifier model')
    parser.add_argument('--C', type=float, default=1e-2, help='classifier model')
    parser.add_argument('--lr_schedule', type=str, default='{0:0.01,25:0.001,50:0.0001}', help='lr schedule for FC')

    args = parser.parse_args()

    J=args.J
    K=0
    if args.dataset=='cifar':
        N_sample_train = 50000
        N_sample_test = 10000

        N=32
    elif args.dataset=='mnist':
        N_sample_train = 60000
        N_sample_test=10000
        N=28
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    feature = None
    if args.feature=='scattering':
        J=args.J
        if args.mode == 1:

            print('order 1')
            feature = Scattering2D(J=args.J, shape=(N, N), max_order=1)
            K = (1+8*J)
        elif args.mode ==2:
            feature = Scattering2D(J=args.J, shape=(N, N))
            K = (1+8*J+32*J*(J-1))

        if use_cuda:
            feature = feature.cuda()
    elif args.feature=='identity':
        feature = nn.Sequential()
        K=1
        J=0
    elif args.feature=='interferometric':
        feature = InterferometricModule(K=args.mode, J=args.J,N=N)
        state = torch.load(args.path)
        feature.W_real.data = state['W_real']
        feature.W_imag.data = state['W_imag']
        K = args.mode+1
    if args.dataset=='cifar':
        K=K*3

    bn = nn.BatchNorm1d(K*(N//(2**J))**2, affine=False).cuda()
    model = nn.Linear(K*(N//(2**J))**2,10).cuda()

    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = None
        pin_memory = False


    if args.dataset=='cifar':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=scattering_datasets.get_dataset_dir('CIFAR'), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    elif args.dataset=='mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data_mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data_mnist', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=128, shuffle=True)

    if args.classifier == 'fc':

        learning_rates = ast.literal_eval(args.lr_schedule)
        # Optimizer
        lr = 0.1 /1000.0
        for epoch in range(0, 90):
            if epoch in learning_rates:
                optimizer = optim.Adam(model.parameters(), lr=learning_rates[epoch])
                print(str(epoch) + ':' + str(learning_rates[epoch]))

            train(model, device, train_loader, optimizer, epoch+1, feature,args.C,bn=bn)
            test(model, device, test_loader, feature,bn=bn)
    elif args.classifier=='svm':
        N=32
        X_train = torch.zeros(N_sample_train ,K*(N//(2**J))**2).float()
        y_train = torch.zeros(N_sample_train).float()
        X_test = torch.zeros(N_sample_test, K * (N // (2 ** J)) ** 2).float()
        y_test = torch.zeros(N_sample_test).float()
        b=0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = feature(data)
            if args.feature != 'scattering':
                output = output[..., 0:N:2 ** J, 0:N:2 ** J].contiguous()
            X_train[b:b+output.size(0),:]=output.view(output.size(0), -1).cpu()
            y_train[b:b + output.size(0)] = target.cpu()
            b+=output.size(0)
        b=0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = feature(data)
            if args.feature != 'scattering':
                output = output[...,0:N:2**J,0:N:2**J].contiguous()
            X_test[b:b + output.size(0), :] = output.view(output.size(0), -1).cpu()
            y_test[b:b + output.size(0)] = target.cpu()
            b += output.size(0)
        acc = SVM(y_train.numpy(), X_train.numpy(), y_test.numpy(), X_test.numpy(), args.C)



if __name__ == '__main__':
    main()
