"""Exponentially weighted (tapered) Pearson correlations over a sliding window."""
import numpy as np


def tapered_weights(window):
    theta = np.round(window / 3)
    w0 = (1 - np.exp(-1 / theta)) / (1 - np.exp(-window / theta))
    return w0 * np.exp(np.arange(-window + 1, 1) / theta)


class TaperedCorrelation:
    """Weighted moments of the current window, updated in O(n^2) per new sample."""

    def __init__(self, block):
        window = len(block)
        weights = tapered_weights(window)
        self.w0, self.lam = weights[-1], weights[-2] / weights[-1]
        self.lam_w = self.lam ** window
        self.a = weights @ block / self.w0
        self.B = (block * weights[:, None]).T @ block / self.w0

    def update(self, x_new, x_old):
        self.a = self.lam * self.a + x_new - self.lam_w * x_old
        self.B = self.lam * self.B + np.outer(x_new, x_new) - self.lam_w * np.outer(x_old, x_old)

    def correlation(self, constant, threshold=0.0):
        mu = self.w0 * self.a
        C = self.w0 * self.B - np.outer(mu, mu)
        C = 0.5 * (C + C.T)
        d = np.sqrt(np.clip(np.diag(C), 0, None))
        with np.errstate(divide='ignore', invalid='ignore'):
            r = C / np.outer(d, d)
        # series that are constant within the window have no defined correlation
        r[constant, :] = 0.0
        r[:, constant] = 0.0
        r = np.nan_to_num(r, nan=0.0)
        np.fill_diagonal(r, 1.0)
        r[np.abs(r) < threshold] = 0.0
        return r


def tapered_correlations(X, window, threshold=0.0):
    """Correlation matrix of every window X[w:w + window], shape (len(X) - window, n, n)."""
    X = np.asarray(X, dtype=float)
    n_windows, n = X.shape[0] - window, X.shape[1]
    if n_windows <= 0:
        return np.empty((0, n, n))
    windows = np.lib.stride_tricks.sliding_window_view(X, window, axis=0)[:n_windows]
    constant = windows.max(axis=2) == windows.min(axis=2)
    stream = TaperedCorrelation(X[:window])
    out = np.empty((n_windows, n, n))
    for w in range(n_windows):
        if w > 0:
            stream.update(X[w + window - 1], X[w - 1])
        out[w] = stream.correlation(constant[w], threshold)
    return out
