import numpy as np
from scipy.signal import lfilter
class referenceModel:

    def __init__(self, tau: float, delay_time: float, ts: float = 1.0):
        self.tau = tau
        self.delay_time = delay_time
        self.ts = ts
        self.d = int(round(delay_time / ts))
        self.a = np.exp(-ts / tau)

    def predict(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        b = [1.0 - self.a]
        a = [1.0, -self.a]
        y_no_delay = lfilter(b, a, r)
        if self.d == 0:
            return y_no_delay
        y = np.zeros_like(y_no_delay)
        if self.d < len(y):
            y[self.d :] = y_no_delay[: -self.d]
        return y

    def inv_predict(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n <= self.d:
            return np.empty(0, dtype=float)
        y_future = y[self.d :]
        y_prev = y[self.d - 1 : -1] if self.d > 0 else np.r_[0.0, y[:-1]]
        return (y_future - self.a * y_prev) / (1.0 - self.a)
