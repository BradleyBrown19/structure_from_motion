"""%% EstimateFundamentalMatrix
% Estimate the fundamental matrix from two image point correspondences 
% Inputs:
%     x1 - size (N x 2) matrix of points in image 1
%     x2 - size (N x 2) matrix of points in image 2, each row corresponding
%       to x1
% Output:
%    F - size (3 x 3) fundamental matrix with rank 2"""
import numpy as np
import pdb

def EstimateFundamentalMatrix(x1, x2):
    A = []

    for p1, p2 in zip(x1,x2):
        A.append(
            np.array( [p1[0]*p2[0], p1[0]*p2[1], p1[0], p1[1]*p2[0], p1[1]*p2[1], p1[1], p2[0], p2[1], 1 ] )[None, :] 
            )

    A = np.concatenate(A)

    U,E,V = np.linalg.svd(A)

    F = V[8].reshape(3,3)

    #Enforce rank 2 constraint
    U,E,V = np.linalg.svd(F)

    E[2] = np.zeros(E[2].shape)

    F = np.matmul(U ,E ,V)

    return np.linalg.norm(F, axis=1, ord=2)

"""%% EssentialMatrixFromFundamentalMatrix
% Use the camera calibration matrix to esimate the Essential matrix
% Inputs:
%     K - size (3 x 3) camera calibration (intrinsics) matrix
%     F - size (3 x 3) fundamental matrix from EstimateFundamentalMatrix
% Outputs:
%     E - size (3 x 3) Essential matrix with singular values (1,1,0)"""

def EssentialMatrixFromFundamentalMatrix(K, F):
    E = K.transpose()*F*K
    U,EP,V = np.linalg.svd(E)
    EP[EP.shape[0]-1] = 0
    E = np.matmul(U ,EP ,V)
    return np.linalg.norm(E, axis=1, ord=2)