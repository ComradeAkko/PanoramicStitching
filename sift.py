# sift.py by Jiro Mizuno

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import math
import os
    
# scale-space extrema detection
def ssExtremaDetect(imgPath):
    img = mpimg.imread(imgPath)
    img = grayScale(img)
    imgs = gaussPyramid(img, 4, 3, 0.5)
    imgs = diffGaussPyramid(imgs)
    candidates = candidatePyramid(imgs)
    keypoints = findPyramidKeypoints(candidates, imgs)

# returns qualified keypoints that are localized, not low contrast, and not on an edge
def findPyramidKeypoints(candidates, pyramid):
    pyraKeys = []
    for i in range(len(pyramid)):
        pyraKeys.append(findOctaveKeypoints(candidates[i], pyramid[i]))

    return pyraKeys

# returns the qualified keypoints of an octave by
# offsetting it, checking for low contrast, and whether it's on the edge

# heavy guidance from the original paper and https://dsp.stackexchange.com/questions/10403/sift-taylor-expansion
# for additional explanations
def findOctaveKeypoints(octCandidates, oct):
    keypoints = []

    # cycle through the candidates of the octave
    for cand in octCandidates:
        x = cand[0]
        y = cand[1]
        z = cand[2]

        imgRow, imgCol = oct[0].shape 

        outofBounds = False

        # iterate at most five times to try to converge to an offset
        for i in range(5):
            # calculate the first derivatives and 2nd derivatives of the candidate
            dx = (oct[z][x+1,y] - oct[z][x-1,y])/2
            dy = (oct[z][x,y+1] - oct[z][x,y-1])/2
            dz = (oct[z+1][x,y] - oct[z-1][x,y])/2

            dxx = oct[z][x+1,y] - 2*oct[z][x,y] + oct[z][x-1,y]
            dxy = ((oct[z][x+1,y+1] - oct[z][x-1,y+1]) - (oct[z][x+1,y-1] - oct[z][x-1,y-1]))/4
            dyy = oct[z][x,y+1] - 2*oct[z][x,y] + oct[z][x,y-1]
            dyz = ((oct[z+1][x,y+1] - oct[z+1][x,y-1]) - (oct[z-1][x,y+1] - oct[z-1][x,y-1]))/4
            dzz = oct[z+1][x,y] - 2*oct[z][x,y] + oct[z-1][x,y]
            dxz = ((oct[z+1][x+1,y] - oct[z+1][x-1,y]) - (oct[z-1][x+1,y] - oct[z-1][x-1,y]))/4

            # construct gradient and hessian
            grad = np.array([dx,dy,dz])
            hessian = np.array([[dxx,dxy,dxz],[dxy,dyy,dyz],[dxz,dyz,dzz]])

            # calculate offset
            offset = -np.linalg.lstsq(hessian, grad, rcond=None)[0]

            # if the offset is less than 0.5 in all dimensions, break because the keypoint is ok
            if offset[0] < 0.5 and offset[1] < 0.5 and offset[2] < 0.5:
                break
            
            # update the keypoint position
            x += int(round(offset[0]))
            y += int(round(offset[1]))
            z += int(round(offset[2]))

            # if the offset is out of bounds, skip it
            if x < 0 or x >= imgRow-1 or y < 0 or y >= imgCol-1 or z < 1 or z >= len(oct)-1:
                outofBounds = True
                break
        
        # if the offset isn't out of bounds continue to further checks
        if not outofBounds:
            # get the function value at the extrenum and check for low contrast
            funcV = oct[z][x,y] + np.dot(grad, offset)/2
            
            # if the function value is greater than or equal to 0.03, check to see if point is on edge
            if funcV >= 0.03:
                # calculate the trace and determinant based on 2x2 Hessian
                trace = dxx + dyy
                det = dxx*dyy - dxy**2

                # if the determinant is positive, continue (non-positive means det is point is not extrenum
                # as the curvature has different sides)
                if det > 0:
                    # check the ratio principal curvature to see if it is below a threshold (not an edge)
                    if trace**2/det < (10+1)**2/10:
                        
                        # add keypoint if it has cleared all conditions
                        keypoints.append([x,y,z])

    return keypoints

# returns the total keypoint candidates of the DoG pyramid
def candidatePyramid(pyramid):
    totalCandidates = []
    for i in range(len(pyramid)):
        totalCandidates.append(candidateOctave(pyramid[i]))
    return totalCandidates

# returns the octave's candidate points
def candidateOctave(oct):
    candidates = []
    imgRow, imgCol = oct[0].shape 
    for i in range(1,len(oct)-1):
        for j in range(1, imgRow-1):
            for k in range(1, imgCol-1):
                # get the maximums and minimums of the above, lower and current layers
                aboveMx = np.max(oct[i+1][j-1:j+2, k-1:k+2])
                aboveMn = np.min(oct[i+1][j-1:j+2, k-1:k+2])
                belowMx = np.max(oct[i-1][j-1:j+2, k-1:k+2])
                belowMn = np.min(oct[i-1][j-1:j+2, k-1:k+2])

                currMx =  np.max(oct[i][j-1:j+2, k-1:k+2])
                currMn =  np.min(oct[i][j-1:j+2, k-1:k+2])

                # if the center pixel is the unchallenged maximum/minimum in above, lower and current layers, add it to candidates
                if max([aboveMx, belowMx, currMx]) == oct[i][j,k] or min([aboveMn, belowMn, currMn]) == oct[i][j,k]:
                    candidates.append([j,k,i])
    return candidates
    
# calculate the difference of gaussians in each octave
def diffGaussPyramid(pyramid):
    diffPyramid = []
    
    # for every octave, calculate the difference in gaussians
    for i in range(len(pyramid)):
        oct = pyramid[i]
        diffOct = []
        for j in range(len(oct)-1):
            diffOct.append(oct[j] - oct[j+1])
        diffPyramid.append(diffOct)

    return diffPyramid

# creates a gaussian pyramid based on the # of octaves and the initial kernel sd
def gaussPyramid(img, oct, s, sd):
    pyramid = []
    currImg = img

    # for every octave:
    #   - generate a series of imgs within the octave
    #   - downsize the img by half and get another octave 
    #       (based on the third last interval image which is 2x the blur)
    for i in range(oct):
        octave = gaussOctave(currImg, s, sd)
        pyramid.append(octave)
        currImg = octave[-3][::2,::2]
    
    return pyramid

# creates a gaussian octave based upon the number of intervals till kernel sd is doubled
def gaussOctave(img, s, sd):
    # start the octave off with the original image
    octave = []

    # the scalar to scale sd via intervals
    k = 2**(1/s)

    # apply and append the first level of blur
    octave.append(convolve(img, kernel(sd)))
    currImg = img
    currSD = k*sd

    # get the remaining s+2 intervals as the paper describes
    # and convolve the original with kernels that are gradually incremented
    # until the blurring is effectively doubled, plus more for later use

    ###############################################################################################
    # WRITE MORE ABOUT WHY +2 MORE INTERVALS FOR LEARNING PURPOSES
    ###############################################################################################
    for i in range(s+2):
        nInterval = convolve(img,  kernel(currSD))
        currSD = math.sqrt((k*currSD)**2 - currSD**2)
        octave.append(nInterval)
        currImg = nInterval
        
    return octave

# creates a gaussian kernel for later use in filtering, input is standard deviation
#   **also a better version from the one created in canny.py, 
#     because it's based on sd instead of rigid size
def kernel(sd):
    lim = math.ceil(3*sd)
    x, y = np.mgrid[-lim:lim+1, -lim:lim+1]
    k = 1/math.sqrt(2*math.pi*sd**2) * np.exp(-(x**2 + y**2)/(2*sd**2))
    return k/np.sum(k)

#########################################
# From canny.py in my EdgeDetection repo
#########################################

# greyscales an matplotlib image by using ITU-R 601-2 luma transformation
# based on https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def grayScale(img):
    return np.dot(img, [0.2989, 0.5870, 0.1140])

# convolves an image based on the filter provided
def convolve(img, filterArray):
    # get the dimensions of the image and the kernel
    imgRow, imgCol = img.shape
    fRow, fCol = filterArray.shape

    # create a padded image with a zero-d border for calculating convolution later more easily
    padImg = np.zeros((imgRow+fRow, imgCol+fCol))
    padImg[fRow//2:imgRow+fRow//2, fCol//2:imgCol+fCol//2] = img

    # color in the padding pixels with the "generally closest" pixel colors so
    # the borders don't get darkened when convolving at the edge
    padImg[:fRow//2, :fCol//2] = img[0,0]
    padImg[:fRow//2, imgCol+fCol//2:] = img[0,imgCol-1]
    padImg[imgRow+fRow//2:,imgCol+fCol//2:] = img[imgRow-1,imgCol-1]
    padImg[imgRow+fRow//2:, :fCol//2] = img[imgRow-1,0]

    for i in range(imgRow):
        padImg[fRow//2 + i, :fCol//2] = img[i,0]
        padImg[fRow//2 + i, imgCol+fCol//2:] = img[i, imgCol-1]
    for i in range(imgCol):
        padImg[:fRow//2, i + fCol//2] = img[0, i]
        padImg[imgRow+fRow//2:, i + fCol//2] = img[imgRow-1, i]

    # initialize the result img
    result = np.zeros((imgRow, imgCol))

    # filter each pixel based on its surroundings
    for i in range(imgRow):
        for j in range(imgCol):
            result[i,j] = np.sum(filterArray * padImg[i:i+fRow, j:j+fCol])
            # result[i,j] = np.sum(filterArray * padImg[i-fRow//2:i+fRow//2, j-fCol//2:j+fCol//2])

    return result

# requires 1 path to the image
if __name__ == '__main__':
    if len(sys.argv) < 3:
        pic = os.getcwd() + "\\" + sys.argv[1]
        ssExtremaDetect(pic)
    else:
        print("Function requires only 1 picture")
        raise SystemExit