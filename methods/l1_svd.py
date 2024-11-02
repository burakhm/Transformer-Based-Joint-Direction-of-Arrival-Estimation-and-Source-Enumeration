import numpy as np
import cvxpy as cp


def l1_svd(received_data, num_signal, steering_matrix, angle_grids, res, reg):
    """L1 norm based sparse representation algorithms for DOA estimation

    Args:
        received_data : Array received signals
        num_signal : Number of signals
        array : Instance of array class
        signal_fre: Signal frequency
        angle_grids : Angle grids corresponding to spatial spectrum. It should
            be a numpy array.
        unit : Unit of angle, 'rad' for radians, 'deg' for degrees.. Defaults to
            'deg'.

    Reference:
        Malioutov, D., M. Cetin, and A.S. Willsky. “A Sparse Signal
        Reconstruction Perspective for Source Localization with Sensor Arrays.”
        IEEE Transactions on Signal Processing 53, no. 8 (August 2005): 3010-22.
        https://doi.org/10.1109/TSP.2005.850882.
    """
    # build the overcomplete basis
    a_over = steering_matrix

    num_samples = received_data.shape[1]

    _, _, vh = np.linalg.svd(received_data)

    d_k = np.vstack(
        (np.eye(num_signal), np.zeros((num_samples - num_signal, num_signal)))
    )
    y_sv = received_data @ vh.conj().transpose() @ d_k

    # solve the l1 norm problem using cvxpy
    p = cp.Variable()
    q = cp.Variable()
    r = cp.Variable(len(angle_grids))
    s_sv = cp.Variable((len(angle_grids), num_signal), complex=True)

    # constraints of the problem
    constraints = [cp.norm(y_sv - a_over @ s_sv, "fro") <= p, cp.sum(r) <= q]
    for i in range(len(angle_grids)):
        constraints.append(cp.norm(s_sv[i, :]) <= r[i])

    # objective function
    objective = cp.Minimize(p + reg * q)
    prob = cp.Problem(objective, constraints)

    prob.solve()

    spectrum = s_sv.value
    spectrum = np.sum(np.abs(spectrum), axis=1)

    pred_ids = get_peaks(spectrum)
    if len(pred_ids) < num_signal:
        pred_ids_temp = np.argsort(spectrum)[::-1]
        pred_ids = np.concatenate((pred_ids,pred_ids_temp[len(pred_ids):num_signal]))

    pred = pred_ids[:num_signal]
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