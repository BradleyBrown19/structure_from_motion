import numpy as np
import pdb

def vec2skew(x):
    if x.shape[0] < 3:
        x = np.append(x,1)
    return np.array([
       [0, -x[2], x[1]],
       [x[2], 0, -x[0]],
       [-x[1], x[0], 0]
    ])