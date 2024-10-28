import numpy as np

def weighted_least_square(B:np.array, L:np.array, P:np.array=None):
    if P is None:
        P = np.eye(L.shape[0])
    L = L.reshape(-1,1)
    X = np.linalg.pinv(B.T @ P @ B) @ (B.T @ P @ L)
    return X