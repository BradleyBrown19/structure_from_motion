"""%% LinearPnP
% Getting pose from 2D-3D correspondences
% Inputs:
%     X - size (N x 3) matrix of 3D points
%     x - size (N x 2) matrix of 2D points whose rows correspond with X
%     K - size (3 x 3) camera calibration (intrinsics) matrix
% Outputs:
%     C - size (3 x 1) pose transation
%     R - size (3 x 1) pose rotation
%
% IMPORTANT NOTE: While theoretically you can use the x directly when solving
% for the P = [R t] matrix then use the K matrix to correct the error, this is
% more numeically unstable, and thus it is better to calibrate the x values
% before the computation of P then extract R and t directly"""
from vec2skew import vec2skew
import numpy as np
import pdb

def LinearPnP(X, x, K):
    x = np.concatenate((x, np.ones((x.shape[0],1))), axis=1)
    X = np.concatenate((x, np.ones((x.shape[0],1))), axis=1)

    xc = np.matmul(np.linalg.inv(K), x.transpose())

    pms = []

    # build 3 x 12 matrix for each points
    for Xp, xp in zip(X,x):
        xp = vec2skew(xp)
        canv = np.zeros((3, 12))

        for i in range(3):
            canv[i][i*4:i*4+4] = Xp

        pms.append(np.matmul(xp,canv))
    
    A = np.concatenate(pms, axis=0)
    U,E,V = np.linalg.svd(A)
    V = V / V[-1,-1]

    P = V[:,-1].reshape(3,4)

    R = P[:,:3]

    U,E,V = np.linalg.svd(R)

    R = np.matmul(U,V)
    t = P[:,3] / E[0]

    if np.linalg.det(R) < 0:
        return np.matmul(R.transpose(),t),R
    else:
        return -np.matmul(R.transpose(),t),-R