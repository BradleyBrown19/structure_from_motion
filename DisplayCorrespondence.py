"""function DisplayCorrespondence(img, x, xp)
%% Display correspondence points between SIFT keypoints and reprojection
% img: image to display
% x: size of (n, 2). SIFT keypoints locations
% xp: size of (n, 2). Reprojection locations"""
import cv2
import numpy as np
import pdb

def DisplayCorrespondence(img, x, xp, idx):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for point in x:
        print(point)
        point = (int(point[0]), int(point[1]))
        cv2.circle(img,tuple(point),3,(0,0,255))
    
    for point in xp:
        point = (int(point[0]), int(point[1]))
        cv2.circle(img,tuple(point),3,(255,0,255))


    cv2.imshow('image' + str(idx),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

