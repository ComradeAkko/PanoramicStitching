# sift.py by Jiro Mizuno

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import math
import os

######################################################################################################
# are the keypoints found at candidatePyramid and orientation using the diffGaussPyramid?
######################################################################################################
    
# scale-space extrema detection
def sift(imgPath):
    sigma = 0.5
    numInterval = 3
    img = mpimg.imread(imgPath)
    img = grayScale(img)
    imgs = gaussPyramid(img, 4, numInterval, sigma)
    imgsDiff = diffGaussPyramid(imgs)
    candidates = candidatePyramid(imgsDiff)
    keypoints = findPyramidKeypoints(candidates, imgsDiff)
    orientedKeys = assignPryaOri(keypoints, imgs, numInterval, sigma)
    print(len(orientedKeys))
    orientedKeys = removeDuplicates(orientedKeys)
    descript = generateDescriptors(orientedKeys, imgs)
    print(len(orientedKeys))

# generates descriptors from keypoints 
def generateDescriptors(keypoints, pyramid, winWidth=16):
    # keypoints in [scaled x, scaled y, scale, octave #, orientation] format
    descriptors = []
    for kp in keypoints:
        # create a windowWidth/4 x windowWidth/4 x 8 vector to store the descriptors (8=number of degree bins)
        # adding 2 to the number of windowWidth bins to account for boundary bins that will be removed later
        # bins are divided into -180~-136, -135~-91,...,270~314,315~360 degrees
        descriptiveVec = np.zeros(shape=(winWidth/4,winWidth/4,8))

        # create a windowWidth * windowWidth to calculate the magnitude and orientation for section of the sample patch
        sampleWindow = np.zeros(shape=(winWidth+1, winWidth+1))
        smoothedSamples = np.zeros(shape = (winWidth+2, winWidth+2, 8))

        # extract the position, scale, octave and orientation of the point 
        xP = kp[0]
        yP = kp[1]
        zP = kp[2]
        octave = kp[3]
        angle = kp[4]
        cosine = math.cos(math.radians(angle))
        sine = math.sin(math.radians(angle))

        # extract the gaussian where the keypoint comes from
        scale = pyramid[octave][zP]
        imgRow, imgCol = scale.shape
    
        # weight the gaussian sigma as half the windowwidth
        halfWidth = winWidth/2
        weightFac = -1/2 * 1/halfWidth**2

        # for every "point" record the partial magnitude and orientation to...
        # ...later be sorted into different bins
        for x in range(-halfWidth, halfWidth+1):
            for y in range(-halfWidth, halfWidth+1):
                xR = round(x * cosine - y * sine + xP)
                yR = round(x * sine + y * cosine + yP)
                
                # only record if the rotated coordinates are within bounds
                # also account for the later magnitude and orientation calculations
                if xR < -1 or xR > imgRow or yR < -1 or yR > imgCol:
                    sampleWindow[x+halfWidth, y+halfWidth] = ("NaN", 0, 0, 0)
                else:
                    # translate the rotation coordinates to the keypoint position
                    xR += xP
                    yR += yP

                    # assign the appropriate, partial bin coordinates
                    xBin = x + halfWidth - 0.5
                    yBin = y + halfWidth - 0.5

                    # assign a gradient magnitude and orientation, while adjusting the orientation to the rotated window
                    gMag = math.sqrt((scale[xR+1,yR] - scale[xR-1,yR])**2 + (scale[xR, yR+1] - scale[xR, yR-1])**2)
                    ori = np.arctan2(scale[xR,yR+1] - scale[xR, yR-1], scale[xR+1,yR] - scale[xR-1,yR])/math.pi * 180 - angle + 180

                    # weight the magnitude by its distance from the keypoint
                    gMag *= math.exp(weightFac * (x**2 + y**2))

                    # get the partial bin # of the orientation
                    binOri = ori/45

                    # record the adjusted magnitude, the orientation, and the bin coordinates
                    sampleWindow[x+halfWidth, y+halfWidth] = (gMag, binOri, xBin, yBin)
        
        # cycle through every sample pixel and use inverse trilinear interpolation to...
        # ...redistribute the magnitudes
        for i in range(winWidth+1):
            for j in range(winWidth+1):
                gMag, binOri, xBin, yBin = sampleWindow[i,j]

                # make sure the sample is valid
                if gMag != "NaN":
                    # get the floor of the bins to extract the fractional proportion between bins
                    binFloor = math.floor(binOri)
                    xFloor = math.floor(xBin)
                    yFloor = math.floor(yBin)

                    binFrac = binOri - binFloor
                    xFrac = xBin - xFloor
                    yFrac = yBin - yFloor

                    c0 = gMag * xFrac
                    c1 = gMag * (1-xFrac)
                    c00 = c0 * yFrac
                    c01 = c0 * (1-yFrac)
                    c10 = c1 * yFrac
                    c11 = c1 * (1-yFrac)
                    c000 = c00 * binFrac
                    c001 = c00 * (1-binFrac)
                    c010 = c01 * binFrac
                    c011 = c01 * (1-binFrac)
                    c100 = c10 * binFrac
                    c101 = c10 * (1-binFrac)
                    c110 = c11 * binFrac
                    c111 = c11 * (1-binFrac)

                    smoothedSamples[i,j,binFloor] += c000
                    smoothedSamples[i,j,binFloor+1] += c001
                    smoothedSamples[i,j+1,binFloor] += c010
                    smoothedSamples[i,j+1,binFloor+1] += c011
                    smoothedSamples[i+1,j,binFloor] += c100
                    smoothedSamples[i+1,j,binFloor+1] += c101
                    smoothedSamples[i+1,j+1,binFloor] += c110
                    smoothedSamples[i+1,j+1,binFloor+1] += c111

        # sum up the smoothed samples into a more concentrated 4x4x8 array
        for a in range(winWidth/4):
            for b in range(winWidth/4):
                for c in range(8):
                    descriptiveVec[a,b,c]  +=  smoothedSamples[a+1,b+1,c] + \
                                            smoothedSamples[a+2,b+1,c] + \
                                            smoothedSamples[a+1,b+2,c] + \
                                            smoothedSamples[a+2,b+2,c]
        
        descriptiveVec = descriptiveVec.flatten()
        descriptiveVec /= np.max(descriptiveVec)
        descriptors.append(descriptiveVec)

    return descriptors


# remove duplicate keypoints
def removeDuplicates(keypoints):
    # sorting multiple attribute arrays: https://stackoverflow.com/questions/31942169/python-sort-array-of-arrays-by-multiple-conditions
    sortKP = lambda x:(x[0], x[1], x[2], x[3], x[4], x[5])
    keypoints.sort(key=sortKP)

    # only take unique keypoints
    unique = [keypoints[0]]

    # compare all the keypoints to make sure none are duplicated
    for nextKP in keypoints[1:]:
        lastKP = unique[-1]
        if  lastKP[0] != nextKP[0] or \
            lastKP[1] != nextKP[1] or \
            lastKP[2] != nextKP[2] or \
            lastKP[3] != nextKP[3] or \
            lastKP[4] != nextKP[4] or \
            lastKP[5] != nextKP[5]:
                unique.append(nextKP)
    
    return unique


# assigns orientations to keypoints
# returns keypoints in [scaled x, scaled y, scale, octave #, orientation, response] format
def assignPryaOri(keypoints, pyramid, s, sd):
    oriKeys = []
    # for each keypoint array for each octave
    for i in range(len(keypoints)):
        for j in range(len(keypoints[i])):
            # extract the point
            point = keypoints[i][j]
            xP = point[0]
            yP = point[1]
            zP = point[2]
            scale = pyramid[i][zP]
            imgRow, imgCol = scale.shape

            # get the adjusted size of the sigma used for the gaussian at the octave interval (the scaled sd*3) scaled the scale factor (1.5)
            k = 2**(1/s)
            currSD = k*sd
            # print(zP)
            for m in range(zP):
                currSD = math.sqrt((k*currSD)**2 - currSD**2)
            
            sigma = round(1.5 * 3*currSD)

            # initialize orientation degree histogram divided into -180 ~ -171,-170 ~ 161,...,170~180 degrees
            histo = [0] * 36

            # for every point within the sigma raidus, add the orientation to the histogram
            for x in range(-sigma, sigma + 1):
                # make sure the point is within bounds to save computing time
                if xP + x > 0 and xP + x < imgRow-1:
                    for y in range(-sigma, sigma + 1):
                        if yP + y > 0 and yP +y < imgCol-1:
                            # assign a gradient magnitude and orientation (converting from -pi to pi radians to 0-360 degrees for bin) according to the paper
                            gMag = math.sqrt((scale[x+1,y] - scale[x-1,y])**2 + (scale[x, y+1] - scale[x, y-1])**2)
                            ori = np.arctan2(scale[x,y+1] - scale[x, y-1], scale[x+1,y] - scale[x-1,y])/math.pi * 180 + 180

                            # the gaussian weight, the constant doesn't really matter because its relative
                            weight = math.exp(-(x**2 + y**2)/(2*currSD**2)) 

                            # get the bin index 
                            bin = math.floor(ori/36)
                            # in the rare case the orientation yields positive 180, change it so it doesn't break the bins
                            if bin == 10:
                                bin = 9

                            # add the weighted magnitude to the bin
                            histo[bin] += weight * gMag
        
            # record the maximum bin peak orientation, and any other orientations that are within 80% of the max peak
            limPeak = max(histo) * 0.8
            for k in range(len(histo)):
                if histo[k] >= limPeak:
                    # interpolate the orientation parabolically and add to the array of keypoints
                    # [scaled x, scaled y, scale, octave #, orientation]
                    oriKeys.append((xP,yP,zP, k, fitParabola(k, histo)))

    return oriKeys

                            

# fits a parabola based on three closest bins close to the input peak and returns the peak
# based on https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
def fitParabola(binIndex, histo):
    # get the right and left
    centVal = 0 
    leftVal = 0
    rightVal = 0
    if binIndex == 0:
        centVal = histo[0]
        leftVal = histo[35]
        rightVal = histo[1]
    elif binIndex == 35:
        centVal = histo[35]
        leftVal = histo[34]
        rightVal = histo[0]
    else:
        centVal = histo[binIndex]
        leftVal = histo[binIndex - 1]
        rightVal = histo[binIndex + 1]
    
    # interpolate the index 
    interloIndex = binIndex + 0.5 * (leftVal - rightVal)/(leftVal - 2 * centVal + rightVal)

    # convert the index into degrees and readjust to -180 to 180 scale
    return interloIndex * 36 - 180
            

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
            if x < 1 or x >= imgRow-1 or y < 1 or y >= imgCol-1 or z < 1 or z >= len(oct)-1:
                outofBounds = True
                break
        
        # if the offset isn't out of bounds continue to further checks
        if not outofBounds:
            # get the function value at the extrenum and check for low contrast
            funcV = oct[z][x,y] + np.dot(grad, offset)/2
            
            # if the function value is greater than or equal to 0.03, check to see if point is on edge
            if abs(funcV) >= 0.03:
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
        sift(pic)
    else:
        print("Function requires only 1 picture")
        raise SystemExit