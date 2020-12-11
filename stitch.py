# stitch.py by by Jiro Mizuno

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import math
import random
import os
import cv2
from sift import *

if __name__ == '__main__':
    if len(sys.argv) < 4:
        None
# stitches two images together as best as they the points align using homemade sift
def stitchTwo(img0, img1):
    siftVec0 = sift(img0)
    siftVec1 = sift(img1)
    # calculate sift points and calculate possible matching pairs of points
    matches = matchPoints(siftVec0, siftVec1)

    # RANSAC a possible homography matrix
    homography = RANSAC(matches, siftVec0, siftVec1)

    img0Row, img0Col, chan0 = img0.shape
    img1Row, img1Col, chan1 = img1.shape

def RANSAC(grayImg1, matches, sift0, sift1, iterMax, eucThres, inThres):

    row, col, chan = grayImg1.shape
    bestHomography = np.zeros(1)
    maxInline = 0

    for i in range(iterMax):
        # select four different random matching pairs
        match0 = matches[random.randrange(len(matches))]
        match1 = matches[random.randrange(len(matches))]
        match2 = matches[random.randrange(len(matches))]
        match3 = matches[random.randrange(len(matches))]

        fourMatches = [match0, match1, match2, match3]
        # compute a homography based on the four random matches
        homography = computeHomography(sift0, sift1, fourMatches)

        inliners = 0

        for j in range(len(matches)):
            ind0 = matches[j][0]
            ind1 = matches[j][1]

            xCoor = sift1[ind1][0]
            yCoor = sift1[ind1][1]

            newLoc = np.zeros(3)
            newLoc[0] = sift0[ind0][0]
            newLoc[1] = sift0[ind0][1]
            newLoc[2] = 1

            transformLoc = np.dot(homography, newLoc.T)

            newX = max(min(round(transformLoc[0]/transformLoc[2]), row-1),0)
            newY = max(min(round(transformLoc[1]/transformLoc[2]), col-1), 0)

            eucD = (grayImg1[xCoor, yCoor, 0] - grayImg1[newX, newY, 0])**2

            if eucD < eucThres:
                inliners += 1

        if inliners/len(matches) >= inThres:
            return homography
        
        elif maxInline < inliners:
            maxInline = inliners
            bestHomography = homography
    
    return homography


def computeHomography(sift0, sift1, matchArray):
    # initialize homography solving matricies
    A = np.zeros(shape=(8,8))
    b = np.zeros(8)

    # fill in the matricies
    for i in range(4):
        b[2*i] = sift1[matchArray[i][1]][0] # x'
        b[2*i + 1] = sift1[matchArray[i][1]][1] # y'

        A[2*i,0] = sift0[matchArray[i][0]][0] # x
        A[2*i,1] = sift0[matchArray[i][0]][0] # y
        A[2*i,2] = 1
        A[2*i,6] = -A[2*i,0] * b[2*i] # -x * x'
        A[2*i,7] = -A[2*i,1] * b[2*i] # -y * x'

        A[2*i+1,3] = b[2*i] # x'
        A[2*i+1,4] = b[2*i + 1]  # y'
        A[2*i+1,5] = 1
        A[2*i+1,6] = -A[2*i,0] * b[2*i+1] # -x * y'
        A[2*i+1,7] = -A[2*i,1] * b[2*i+1] # -y * y'

    # solve Ah = b and form homography matrix H
    q, r= np.linalg.qr(A)
    p = np.dot(q.T, b)
    h = np.dot(np.linalg.inv(r), p)

    H = np.zeros(shape=(3,3))
    H[0,0] = h[0]
    H[0,1] = h[1]
    H[0,2] = h[2]
    H[1,0] = h[3]
    H[1,1] = h[4]
    H[1,2] = h[5]
    H[2,0] = h[6]
    H[2,1] = h[7]
    H[2,2] = 1


    return H

# returns an array of tuples containing possibly matching points by brute force matching every possible match's indicies
def matchPoints(sift0, sift1, thres = 1.6):
    matches = []
    for a in range(len(sift0)):
        for b in range(len(sift1)):
            d0 = sift0[a][2]
            d1 = sift1[b][2]

            eucD = 0
            # cacluate the euclidean difference
            for c in range(len(d0)):
                eucD += (d0[c] - d1[c])**2
            eucD = math.sqrt(eucD)

            # if the euclidean difference is below threshold, add the pair to the array of match indicies
            if eucD < thres:
                matches.append((a, b))

    return matches