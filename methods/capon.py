import numpy as np
from scipy import linalg


def get_peaks(spectrum):
    spectrum_temp = np.concatenate([[0],spectrum,[0]])
    crossings = ((np.diff(np.sign(np.diff(spectrum_temp))) == -2)*1)
    crossings = np.where(crossings==1)[0]
    crossing_spectrum = spectrum[crossings]
    crossings_sub = np.argsort(crossing_spectrum)[::-1]
    peaks = crossings[crossings_sub]

    return peaks

def capon(R,M,N,sensor_pos,res):
    phi = np.arange(30, 150+res, res)
    theta = 90
    spectrum = np.zeros(phi.shape[0]);

    for i in range(phi.shape[0]):
        a = np.exp(-2j*np.pi*(sensor_pos[:,0] * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi[i])) + 
                              sensor_pos[:,1] * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi[i])) + 
                              sensor_pos[:,2] * np.cos(np.deg2rad(theta)))).reshape(-1,1)

        spectrum[i] = 1/np.abs(a.conj().T@linalg.inv(R)@a)

    pred_ids = get_peaks(spectrum)
    if len(pred_ids) < N:
        pred_ids_temp = np.argsort(spectrum)[::-1]
        pred_ids = np.concatenate((pred_ids,pred_ids_temp[len(pred_ids):N]))

    pred = pred_ids[:N]
    pred = pred*res + 30
    
    return pred, spectrum