import numpy as np


def svt(N, L, R, sensor_pos):
    delta = 1.2 * (N**2) / L
    mu = 10000
    epsilon = 1e-4
    kmax = 10000

    C = np.zeros(16, dtype="complex")
    counter = np.zeros(16)
    Rhat = np.zeros((16,16), dtype="complex")
    P = np.zeros((16,16))

    for i in range(sensor_pos.shape[0]):
        for j in range(i, sensor_pos.shape[0]):
            idx = int(2*(sensor_pos[j,0]-sensor_pos[i,0]))
            C[idx] += R[i,j]
            counter[idx] += 1
            for k in range(P.shape[0]):
                for n in range(P.shape[1]):
                    if abs(k-n) == idx:
                        P[k,n] = 1

    for i in range(C.shape[0]):
        if counter[i] != 0:
            C[i] /= counter[i]

    for i in range(Rhat.shape[0]):
        for j in range(Rhat.shape[1]):
            if i < j:
                Rhat[i,j] = C[abs(int(j-i))]
            else:
                Rhat[i,j] = C[abs(int(j-i))].conj()

    k0 = np.ceil(np.real(mu/(delta*np.linalg.norm(P*Rhat, ord=2))))
    Y = k0*delta*P*Rhat

    for k in range(kmax):
        U, S, Vh = np.linalg.svd(Y, hermitian=True, full_matrices=True)
        S[S<mu] = 0
        S[S>mu] -= mu
        Rc = U @ np.diag(S) @  Vh
        Y = Y + delta * P * (Rhat-Rc)

        error = np.linalg.norm(P * (Rc-Rhat)) / np.linalg.norm(P * Rhat)
        
        if error <= epsilon:
            break
      
    return Rc