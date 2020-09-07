# Get Dataset
from scipy.io import loadmat
import pdb
import numpy as np

from DisplayCorrespondence import DisplayCorrespondence
from LinearPnP import LinearPnP
from triangulation import LinearTriangulation
from epipolar_correlations import EstimateFundamentalMatrix, EssentialMatrixFromFundamentalMatrix

data = loadmat('data.mat')

x1 = np.array(data['data'][0][0][0])
x2 = np.array(data['data'][0][0][1])
x3 = np.array(data['data'][0][0][2])

img1 = np.array(data['data'][0][0][6])
img2 = np.array(data['data'][0][0][7])
img3 = np.array(data['data'][0][0][8])

# Load camera calibration parameters
K = np.array(data['data'][0][0][5])

# Estimate fundamental matrix
F = EstimateFundamentalMatrix(x1, x2)

# Estimate essential matrix from fundamental matrix
E = EssentialMatrixFromFundamentalMatrix(F,K)

C = data['data'][0][0][4]
R = data['data'][0][0][3]

# Obtain 3d points using correct camera pose
X = LinearTriangulation(K, np.zeros((3,1)), np.eye(3), C, R, x1, x2)

# Find the third camera pose using Linear PnP
C3, R3 = LinearPnP(X, x3, K)


X = X[:,:3]

x1p = np.matmul(K, X.transpose())
x1p = x1p / x1p[2,:]

x2p = X.transpose() - C
x2p = np.matmul(np.matmul(K, R), x2p)
x2p = x2p / x2p[2,:]

x3p = X.transpose() - np.expand_dims(C3, 1)
x3p = np.matmul(np.matmul(K, R3), x3p)
x3p = x3p / x3p[2,:]

#Display correspondence points between SIFT keypoints and reprojection
DisplayCorrespondence(img1, x1, x1p[:3,:].transpose(),1)
DisplayCorrespondence(img2, x2, x2p[:3,:].transpose(),2)
DisplayCorrespondence(img3, x3, x3p[:3,:].transpose(),3)

"""
% Nonlinear triangulation
X = Nonlinear_Triangulation(K, zeros(3,1), eye(3), C, R, C3, R3, x1, x2, x3, X);

% Display point cloud and 3 camera poses
Display3D({zeros(3,1), C, C3}, {eye(3), R, R3}, X);

% Calculate reprojection points
x1p = K * X';
x1p = x1p ./ repmat(x1p(3, :), [3, 1]);
x2p = K * R * (X' - repmat(C, [1 size(X,1)]));
x2p = x2p ./ repmat(x2p(3, :), [3, 1]);
x3p = K * R3 * (X' - repmat(C3, [1 size(X,1)]));
x3p = x3p ./ repmat(x3p(3, :), [3, 1]);

% Display correspondence points between SIFT keypoints and reprojection
DisplayCorrespondence(data.img1, x1, x1p(1:2,:)');
DisplayCorrespondence(data.img2, x2, x2p(1:2,:)');
DisplayCorrespondence(data.img3, x3, x3p(1:2,:)');
"""