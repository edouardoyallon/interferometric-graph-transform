import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


data=torch.load('W.t7',map_location={'cuda:0': 'cpu'})
W_real = data['W_real'].detach()
W_imag = data['W_imag'].detach()
D = np.fft.fftshift(torch.sqrt(W_real**2+W_imag**2).numpy(),axes=(1,2,))
P = torch.from_numpy(D)
P = P/P.max()
z=[]
for i in range(P.size(0)):
    z.append(P[i,...].unsqueeze(0).expand(3,32,32))
x = torchvision.utils.make_grid(z,pad_value=1)#255).detach()
# transpose numpy array to the PIL format, i.e., Channels x W x H
print(x.size())
plt.imshow(np.transpose(x, (1,2,0)), interpolation='nearest')
plt.imsave('W.pdf',np.transpose(x, (1,2,0)))

