import numpy as np


def mode_omp(data, sensor_pos, M, snr):
    source_phi = np.arange(30,151).reshape(-1,1)
    source_the = 90*np.ones((121,1))
    A = np.exp(-2j*np.pi*(sensor_pos[:,0] * np.cos(np.deg2rad(source_phi)) * np.sin(np.deg2rad(source_the)) +
                          sensor_pos[:,1] * np.sin(np.deg2rad(source_phi)) * np.sin(np.deg2rad(source_the)) +
                          sensor_pos[:,2] * np.cos(np.deg2rad(source_the))
                          )).T
    R = data.copy()
    colids = []
    threshold = np.sqrt(M) / np.sqrt(10**(snr/10))
    R_norm = 1e9

    while R_norm > threshold:
        cors = np.sqrt(np.sum(np.abs(A.conj().T @ R)**2, axis=1))
        selected_colid = np.argmax(cors)

        if selected_colid not in colids:
            colids.append(selected_colid)

        U = A[:, colids]
        P = np.linalg.inv(U.conj().T @ U) @ U.conj().T @ data
        R = data - U @ P

        R_norm = np.sqrt(np.sum(np.mean(np.abs(R),axis=1)**2))

    return np.array(colids) + 30