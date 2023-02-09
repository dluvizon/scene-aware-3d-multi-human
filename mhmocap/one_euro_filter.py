# Source code borrowed from: https://github.com/mkocabas/VIBE

import math
import numpy as np


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=None, min_cutoff=0.004, beta=0.7, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        if dx0 is None:
            dx0 = np.zeros_like(x0)
        else:
            assert dx0.shape == x0.shape, (f'Invalid dx0 shape ({dx0.shape})!')
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x, mask=None):
        """Compute the filtered signal."""
        if mask is None:
            mask = np.ones_like(x)
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = (1 - mask) * self.x_prev + (mask * x_hat)
        self.dx_prev = (1 - mask) * self.dx_prev + (mask * dx_hat)
        self.t_prev = (1 - mask) * self.t_prev + (mask * t)

        return (1 - mask) * x + (mask * x_hat)
