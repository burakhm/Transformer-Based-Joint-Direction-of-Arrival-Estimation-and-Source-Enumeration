import numpy as np
from numpy import linalg as la


def t_gane(evs, sel_t):
    threshold = sel_t
    evs = evs / sum(evs)
    M = len(evs)
    N = 1
    
    for j in range(len(evs)-1):
        delta_ev = evs[j+1]-evs[j]
        
        if delta_ev > threshold:
            N = M-j-1
            return N
            
    return N

def music(R,M,N_max,T,sensor_pos,res, sel_t):
    D, V = la.eigh(R)
    idx = abs(D).argsort()   
    D = D[idx]
    V = V[:,idx]
    N = t_gane(D, sel_t)
    noise_V = V[:,:M-N] # noise subspace of R

    phi = np.arange(30, 150+res, res)
    theta = 90
    spectrum = np.zeros(phi.shape[0])

    for i in range(phi.shape[0]):
        a = np.exp(-2j*np.pi*(sensor_pos[:,0] * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi[i])) + 
                              sensor_pos[:,1] * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi[i])) + 
                              sensor_pos[:,2] * np.cos(np.deg2rad(theta)))).reshape(-1,1)
        
        spectrum[i] = 1/(la.norm(a.conj().T@noise_V)**2)
    
    pred_ids = get_peaks(spectrum)
    if len(pred_ids) < N:
        pred_ids_temp = np.argsort(spectrum)[::-1]
        pred_ids = np.concatenate((pred_ids,pred_ids_temp[len(pred_ids):N]))

    pred = pred_ids[:N]
    pred = pred*res + 30
    
    return pred, spectrum

def get_peaks(spectrum):
    spectrum_temp = np.concatenate([[0],spectrum,[0]])
    crossings = ((np.diff(np.sign(np.diff(spectrum_temp))) == -2)*1)
    crossings = np.where(crossings==1)[0]
    crossing_spectrum = spectrum[crossings]
    crossings_sub = np.argsort(crossing_spectrum)[::-1]
    peaks = crossings[crossings_sub]

    return peaks