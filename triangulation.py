"""%% LinearTriangulation
% Find 3D positions of the point correspondences using the relative
% position of one camera from another
% Inputs:
%     C1 - size (3 x 1) translation of the first camera pose
%     R1 - size (3 x 3) rotation of the first camera pose
%     C2 - size (3 x 1) translation of the second camera
%     R2 - size (3 x 3) rotation of the second camera pose
%     x1 - size (N x 2) matrix of points in image 1
%     x2 - size (N x 2) matrix of points in image 2, each row corresponding
%       to x1
% Outputs: 
%     X - size (N x 3) matrix whos rows represent the 3D triangulated
%       points"""
import pdb
import numpy as np
from vec2skew import vec2skew

def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    P1 = np.matmul(K,np.concatenate((C1,R1), axis=1))
    P2 = np.matmul(K,np.concatenate((C2,R2), axis=1))

    x1 = [vec2skew(x) for x in x1]
    x2 = [vec2skew(x) for x in x2]

    pts = []

    for p1,p2 in zip(x1,x2):
        A = np.concatenate((np.matmul(p1, P2), np.matmul(p1, P2)), axis=0)
        U,E,V = np.linalg.svd(A)
        pt = np.expand_dims((V[:,-1] / V[-1,-1]),0)
        pts.append(pt)
        
    return np.concatenate(pts, axis=0)