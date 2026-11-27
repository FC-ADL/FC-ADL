"""DeltaCon affinities and landmark MDS."""
import numpy as np
import scipy.linalg


def affinity_sqrt(A, eps=1e-2):
    """Element-wise sqrt of S = (I + eps^2 D - eps A)^-1 for a stack of adjacency matrices (T, n, n)."""
    A = np.asarray(A, dtype=float)
    I = np.eye(A.shape[1])
    M = I[None] + (eps ** 2) * A.sum(axis=2)[:, :, None] * I[None] - eps * A
    return np.sqrt(np.abs(np.linalg.inv(M)))


def landmarks(n_points, fraction):
    return list(np.linspace(0, n_points - 1, max(3, int(n_points * fraction)), dtype=int))


def lmds_fit(D, lands, dim):
    """Embed all points from their distances to the landmarks, D with shape (n_landmarks, n_points).

    Returns the coordinates and the transform used by lmds_project. Each axis is oriented so that
    its largest-magnitude coordinate is positive, as eigenvector signs are otherwise arbitrary.
    """
    Dl = D[:, lands]
    n = len(Dl)
    H = -np.ones((n, n)) / n
    np.fill_diagonal(H, 1 - 1 / n)
    H = -H.dot(Dl ** 2).dot(H) / 2
    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    w = evals.argsort()[-dim:][::-1]
    if np.any(evals[w] < 0):
        raise ValueError('not enough positive eigenvalues for the selected dimension')
    Lh = evecs[:, w].dot(np.diag(1. / np.sqrt(evals[w]))).T
    N = D.shape[1]
    land_mean = np.mean(Dl, axis=1)
    X = -Lh.dot(D - np.tile(land_mean, (N, 1)).T) / 2.
    x_mean = np.mean(X, axis=1)
    X -= np.tile(x_mean, (N, 1)).T
    _, evecs = scipy.linalg.eigh(X.dot(X.T))
    R = evecs[:, ::-1]
    coords = (R.T.dot(X)).T
    signs = np.sign(coords[np.argmax(np.abs(coords), axis=0), np.arange(coords.shape[1])])
    signs[signs == 0] = 1
    return coords * signs, dict(Lh=Lh, land_mean=land_mean, x_mean=x_mean, R=R, signs=signs)


def lmds_project(model, d_new):
    """Embed new points from their distances to the landmarks, d_new with shape (n_points, n_landmarks)."""
    d_new = np.atleast_2d(d_new)
    X = -model['Lh'].dot(d_new.T - model['land_mean'][:, None]) / 2.
    X -= model['x_mean'][:, None]
    return (model['R'].T.dot(X)).T * model['signs']
