import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sfx = np.exp(x) / np.sum(np.exp(x), axis=0)
    return sfx


def delta_softmax(x, dx):
    _dx = dx * np.log(np.sum(np.exp(x)))
    sfx = np.exp(x - _dx) / np.sum(np.exp(x - _dx))
    return sfx


# ----------------
scores = np.array([3, 1, -2])
dx_scores = np.array([0, 1.0, 0])
print(softmax(scores))
print(delta_softmax(x=scores, dx=dx_scores))
