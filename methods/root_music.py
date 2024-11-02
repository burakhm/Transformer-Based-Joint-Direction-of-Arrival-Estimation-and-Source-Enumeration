import numpy as np
from numpy import linalg as la


def root_music(R,M,N):
    D, V = la.eig(R)
    idx = abs(D).argsort()
    D = D[idx]
    V = V[:,idx]
    noise_V = V[:,:M-N] # noise subspace of R

    C = noise_V @ noise_V.T.conj()
    coeff = np.zeros((2*M-1,), dtype=np.complex_)
    for i in range(-M+1, M):
        coeff[i+M-1] = np.sum(np.diag(C, i))
    z = np.roots(coeff)
    z = z[abs(z)<1]
    idx = (1-abs(z)).argsort()
    idx = idx[:N]
    z_source = z[idx]
    pred = np.arccos(np.angle(z_source)/np.pi)
    pred = np.rad2deg(np.mod(pred, np.pi))

    return pred