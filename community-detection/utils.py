import numpy as np
import torch.nn as nn
import torch
from numpy import linalg as LA
import networkx as nx
import scipy
class InterferometricModule(nn.Module):

    def __init__(self, K=10,  J=3, N=32): #, avg='gaussian'):

        super(InterferometricModule, self).__init__()
        self.W_real = nn.Parameter(torch.randn(K,N,N).squeeze()*torch.bernoulli(0.9*torch.ones(K,N,N)))
        self.W_imag = nn.Parameter(torch.randn(K,N,N).squeeze()*torch.bernoulli(0.5*torch.ones(K,N,N)))

        self.avg = GaussianConv(N=N,J=J).cuda()

        self.K=K
        self.N=N

    def projection_l2_ball(self):
        # First we consider \sum_n |\hat W_n|^2
        W_norm = torch.sum(self.W_real**2 + self.W_imag**2, 0)

        # Then we consider \sum_n |\hat W_n(omega)|^2+|\hat W_n(-omega)|^2 which should be less <=2
        N= W_norm.shape[1]
        for i in range(1,N//2+1):
            if i<N//2:
                for j in range(1,N):
                    W_norm[i,j]=W_norm[i,j]+W_norm[N-i,N-j]
                    W_norm[N-i,N-j]=W_norm[i,j]
            elif i==N/2:
                for j in range(1, N//2+1):
                    W_norm[i, j]=W_norm[i,j]+W_norm[N-i,N-j]
                    W_norm[N - i, N - j] = W_norm[i, j]
        for i in range(1,N//2+1):
            W_norm[0,i]+=W_norm[0,N-i]
            W_norm[0,N-i]=W_norm[0,i]
            W_norm[i,0]+=W_norm[N-i,0]
            W_norm[N-i,0]=W_norm[i,0]
        W_norm[0, 0] = W_norm[0, 0] * 2

        # target
        W_target = torch.sqrt(-2*torch.abs(self.avg.phi_signal)**2 +2)

        # We normalize such that the projection leads to <=2
        W_norm = torch.sqrt(W_norm)
        a = torch.max(W_norm)
        W_norm [W_norm<W_target]=1
        tmp = torch.div(W_target,W_norm)
        W_norm[W_norm>=W_target]=tmp[W_norm>=W_target]
        # We project!
        self.W_real.data *= W_norm.unsqueeze(0).expand(self.W_real.size(0),-1,-1)
        self.W_imag.data *= W_norm.unsqueeze(0).expand(self.W_real.size(0),-1,-1)
        return a


    def forward(self, x):
        N=self.N
        batch=x.size(0)
        x_LF=self.avg(x)
        x = x.view(-1,self.N,self.N) # This step is specific to images - we apply independantly on each channels the transform
        
        x_f = torch.rfft(x,2,onesided=False,normalized=True)
        y_f_r = x_f.new(x_f.size(0),self.K,N,N)
        y_f_i = x_f.new(x_f.size(0),self.K,N,N)
        for k in range(self.K):
            a_r=x_f[...,0].view(x_f.size(0),N,N)
            a_i=x_f[...,1].view(x_f.size(0),N,N)
            b_i= self.W_imag[k,...].unsqueeze(0).expand(x.size(0),-1,-1).squeeze()
            b_r= self.W_real[k,...].unsqueeze(0).expand(x.size(0),-1,-1).squeeze()
            y_f_r[:,k,:,:] = a_r * b_r-a_i*b_i
            y_f_i[:,k,:,:] = a_r*b_i+a_i*b_r
        y_f = torch.cat([y_f_r.unsqueeze(-1),y_f_i.unsqueeze(-1)],-1)
        y = torch.ifft(y_f, 2,normalized=True)

        # Compute modulus
        y = y ** 2
        y = y.sum(-1)
        y = torch.sqrt(y)

        # Average
        y = self.avg(y)
        y = y.view(batch,-1,y.size(2),y.size(3))
        
        return torch.cat([y,x_LF],1)



def gabor_2d(M, N, sigma, theta, xi, slant=1.0, offset=0):
    """
        Computes a 2D Gabor filter.
        A Gabor filter is defined by the following formula in space:
        psi(u) = g_{sigma}(u) e^(i xi^T u)
        where g_{sigma} is a Gaussian envelope and xi is a frequency.
        Parameters
        ----------
        M, N : int
            spatial sizes
        sigma : float
            bandwidth parameter
        xi : float
            central frequency (in [0, 1])
        theta : float
            angle in [0, pi]
        slant : float, optional
            parameter which guides the elipsoidal shape of the morlet
        offset : int, optional
            offset by which the signal starts
        Returns
        -------
        morlet_fft : ndarray
            numpy array of size (M, N)
    """
    gab = np.zeros((M, N), np.complex64)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float32)
    D = np.array([[1, 0], [0, slant * slant]])
    curv = np.dot(R, np.dot(D, R_inv)) / ( 2 * sigma * sigma)

    for ex in [-2, -1, 0, 1, 2]:
        for ey in [-2, -1, 0, 1, 2]:
            [xx, yy] = np.mgrid[offset + ex * M:offset + M + ex * M, offset + ey * N:offset + N + ey * N]
            arg = -(curv[0, 0] * np.multiply(xx, xx) + (curv[0, 1] + curv[1, 0]) * np.multiply(xx, yy) + curv[
                1, 1] * np.multiply(yy, yy)) + 1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
            gab += np.exp(arg)

    gab = np.fft.fft2(gab, norm=None)
    norm_factor = np.amax(np.abs(gab))
    gab /= norm_factor
    return gab

class GaussianConv(nn.Module):

    def __init__(self,N=32,J=3):
        super(GaussianConv, self).__init__()
        self.phi_signal = torch.from_numpy(np.real(gabor_2d(N,N, 0.8 * 2**(J-1), 0, 0))).cuda()

    def forward(self, x):
        x_f = torch.rfft(x,2,normalized=True,onesided=False)
        y_f_r=x_f[...,0]*self.phi_signal.expand_as(x_f[...,0])
        y_f_i=x_f[...,1]*self.phi_signal.expand_as(x_f[...,0])
        y_f = torch.cat([y_f_r.unsqueeze(-1),y_f_i.unsqueeze(-1)],-1)
        return torch.irfft(y_f,2,normalized=True,onesided=False)


def Fourier_atoms(adj, size_avg):
    def adjacencyToLaplacian(W):
        """
        adjacencyToLaplacian: Computes the Laplacian from an Adjacency matrix
        Input:
            W (np.array): adjacency matrix
        Output:
            L (np.array): Laplacian matrix
        """
        # Check that the matrix is square
        assert W.shape[0] == W.shape[1]
        # Compute the degree vector
        d = np.sum(W, axis=1)
        # And build the degree matrix
        D = np.diag(d)
        # Return the Laplacian
        return D - W

    def normalizeLaplacian(L):
        """
        NormalizeLaplacian: Computes the degree-normalized Laplacian matrix
        Input:
            L (np.array): Laplacian matrix
        Output:
            normL (np.array): degree-normalized Laplacian matrix
        """
        # Check that the matrix is square
        assert L.shape[0] == L.shape[1]
        # Compute the degree vector (diagonal elements of L)
        d = np.diag(L)
        # Invert the square root of the degree
        d = 1 / np.sqrt(d)
        # And build the square root inverse degree matrix
        D = np.diag(d)
        # Return the Normalized Laplacian
        return D @ L @ D

    def blossom_hilbert(s):

        FG = nx.Graph()
        ss = s
        for i in range(s.shape[0]):
            for j in range(s.shape[0]):
                if i != j:  # This condition avoid to match the same nodes
                    weight = np.sum(np.abs(ss[j,:] + 1.0j * ss[ i,:]))
                    FG.add_weighted_edges_from([(i, j, weight)])

        mate = nx.max_weight_matching(
            FG)  # A matching is a subset of edges in which no node occurs more than once. The cardinality of a matching is the number of matched edges. The weight of a matching is the sum of the weights of its edges.
        # import ipdb;ipdb.set_trace()
        return mate





    def Fourrier_Lap_atoms(W,size_avg):
        uu, ss, vv = np.linalg.svd(W, full_matrices=False)
        
        A = vv[-size_avg:,:]
        vv=vv[:-size_avg,:]
        uu=uu[:-size_avg,:]
        SIZE = vv.shape[0]
        dim = vv.shape[0]

        mate = blossom_hilbert(vv)
        mate = list(mate)

        FF = np.zeros_like(vv)
        FF = FF.astype(np.complex64)
        j=0
        for i in range( SIZE// 2):
            mm, nn = mate[i][0], mate[i][1]#X[2*i], X[2*i+1]#
            aa = vv[ mm,:]
            bb = vv[nn,:]
            if np.linalg.norm(aa)==0 or np.linalg.norm(bb)==0:
                print('Should not happen')
                aa*=np.sqrt(2)
                bb*=np.sqrt(2)
            FF[j,:] = (aa + 1.0j * bb) / np.sqrt(2)
            FF[ j+1,:] = (aa - 1.0j * bb) / np.sqrt(2)
            j+=2

        FF_conj_tran = FF.conj().transpose()
        Z=FF_conj_tran.dot(FF)

        Vect0 = 0
        return FF, FF_conj_tran, Vect0, A



    norm_adj = LA.norm(adj)
    adj = adj / norm_adj
    lap = adjacencyToLaplacian(adj)
    norm_lap = normalizeLaplacian(lap)
    if size_avg==2:
        FF1, Fc1, Vect01, A1 = Fourrier_Lap_atoms(norm_lap[0:15,0:15], 1)  # Fourier atoms and conjugate
        FF2, Fc2, Vect02, A2 = Fourrier_Lap_atoms(norm_lap[15:, 15:], 1)  # Fourier atoms and conjugate
        FF = scipy.linalg.block_diag(FF1,FF2)
        Fc = scipy.linalg.block_diag(Fc1, Fc2)
        A = np.zeros((2, 30))
        Vect0=Vect01
        A[0, :15] = A1
        A[1, 15:] = A2
    elif size_avg==1:
        FF,Fc,Vect0, A = Fourrier_Lap_atoms(norm_lap,1) # Fourier atoms and conjugate
    return adj,FF,Fc,Vect0, A


class InterferometricModuleGraphs(nn.Module):
    def __init__(self, avg, F, N, K=10 , size_avg=1):  # , avg='gaussian'):

        super(InterferometricModuleGraphs, self).__init__()
        N_init=N-size_avg
        self.W_real = nn.Parameter(torch.randn(K, N_init).squeeze() * torch.bernoulli(0.5 * torch.ones(K, N_init)))
        self.W_imag = nn.Parameter(torch.randn(K, N_init).squeeze() * torch.bernoulli(0.5 * torch.ones(K, N_init)))
        self.avg = torch.from_numpy(avg).float() # first the diagonal case - here assume avg is a vector
        self.F_r=torch.from_numpy(np.real(F))
        self.F_i = torch.from_numpy(np.imag(F))

        self.F_inv_r = torch.from_numpy(np.real(np.conj(F.T)))
        self.F_inv_i = torch.from_numpy(np.imag(np.conj(F.T)))

        self.K = K
        self.N = N
        self.size_avg = size_avg

    def projection_l2_ball(self):
        # First we consider \sum_n |\hat W_n|^2
        W_norm = torch.sum(self.W_real ** 2 + self.W_imag ** 2, 0)


        # Then we consider \sum_n |\hat W_n(omega)|^2+|\hat W_n(-omega)|^2 which should be less <=2
        N = W_norm.shape[0]
        for i in range(0, N // 2 ):
            W_norm[2*i]+=W_norm[2*i+1]
            W_norm[2 * i +1 ]=W_norm[2*i]
        # We normalize such that the projection leads to <=2
        W_norm = torch.sqrt(W_norm)
        if N % 2 == 1:
            W_norm[-1] *= np.sqrt(2)
        W_norm[W_norm < np.sqrt(2)] = 1
        W_norm[W_norm >= np.sqrt(2)] *= 1 / np.sqrt(2)


        # We project!
        self.W_real.data /= W_norm.unsqueeze(0).expand(self.W_real.size(0),  -1)
        self.W_imag.data /= W_norm.unsqueeze(0).expand(self.W_real.size(0),  -1)
        return torch.max(W_norm)

    def forward(self, x, only_averaging=False, no_averaging=False):
        batch1 = x.size(0)
        batch2 = x.size(1)

        x = x.view((-1,self.N))
        if not no_averaging:
            x_LF = x @ self.avg.T
        else:
            x_LF=x
        x_LF = x_LF.view((batch1,batch2,-1))
        if only_averaging:
            return x_LF

        x_f_r = x @ self.F_r.T
        x_f_i = x @ self.F_i.T
        N_init = self.N-self.size_avg

        y_f_r = x_f_r.new(x_f_r.size(0), self.K, N_init)
        y_f_i = x_f_i.new(x_f_i.size(0), self.K, N_init)

        for k in range(self.K):
            a_r = x_f_r
            a_i = x_f_i
            b_i = self.W_imag[k, ...].unsqueeze(0).expand(x.size(0), -1, -1).squeeze()
            b_r = self.W_real[k, ...].unsqueeze(0).expand(x.size(0), -1, -1).squeeze()
            y_f_r[:, k, :] = a_r * b_r - a_i * b_i
            y_f_i[:, k, :] = a_r * b_i + a_i * b_r
        y_f_r = y_f_r.view(-1, N_init)
        y_f_i = y_f_i.view(-1, N_init)
        y_r = y_f_r @ self.F_inv_r.T- y_f_i @ self.F_inv_i.T
        y_i = y_f_i @ self.F_inv_r.T + y_f_r @ self.F_inv_i.T

        # Compute modulus
        y_r = y_r ** 2
        y_i = y_i ** 2
        y = y_r + y_i
        y = torch.sqrt(y)

        y = y.view(batch1, -1, self.N)
        # Average
        return y, x_LF
