# sift.py by Jiro Mizuno

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import math
import os
import cv2

######################################################################################################
# are the keypoints found at candidatePyramid and orientation using the diffGaussPyramid?
######################################################################################################

def showSpots(img0,siftVec):
    img0Row, img0Col, chan0 = img0.shape
    img = np.zeros(shape=(img0Row, img0Col, chan0))
    for i in range(img0Row):
        for j in range(img0Col):
            img[i,j,0] = img0[i,j,0]/255
            img[i,j,1] = img0[i,j,1]/255
            img[i,j,2] = img0[i,j,2]/255    
    imgplot = plt.imshow(img)
    plt.show()
    for i in range(len(siftVec)):
        for j in range(1):
            for k in range(1):
                img[min(max((siftVec[i][0]+j), 0), img0Row-1), min(max((siftVec[i][1]+k), 0), img0Col-1), 0] = 1
                img[min(max((siftVec[i][0]+j), 0), img0Row-1), min(max((siftVec[i][1]+k), 0), img0Col-1), 1] = 0
                img[min(max((siftVec[i][0]+j), 0), img0Row-1), min(max((siftVec[i][1]+k), 0), img0Col-1), 2] = 0

    imgplot = plt.imshow(img)
    plt.show()

# compares the sift points of two images based on a threshold
def compareTwo(img0, img1, siftVec0, siftVec1, threshold = 1.6):
    img0Row, img0Col, chan0 = img0.shape
    img1Row, img1Col, chan1 = img1.shape
    
    comp = np.zeros(shape=(max(img0Row, img1Row), img0Col + img1Col, 3))
    for i in range(img0Row):
        for j in range(img0Col):
            for k in range(3):
                comp[i,j,k] = img0[i,j,k]/255

    for i in range(img1Row):
        for j in range(img1Col):
            for k in range(3):
                comp[i,j+img0Col,k] = img1[i,j,k]/255
    count = 0
    af = 0
    print("starting comparisons")
    print(len(siftVec0))
    print(len(siftVec1))
    eucList = []
    summer = len(siftVec0) * len(siftVec1)
    for a in range(len(siftVec0)):
        for b in range(len(siftVec1)):
            d0 = siftVec0[a][2]
            d1 = siftVec1[b][2]

            eucD = 0
            for c in range(len(d0)):
                eucD += (d0[c] - d1[c])**2
            eucD = math.sqrt(eucD)
            eucList.append(eucD)

            af += 1
            # print(af/summer)

            if eucD < threshold:
                count += 1
                x0 = siftVec0[a][0]
                y0 = siftVec0[a][1]
                x1 = siftVec1[b][0]
                y1 = siftVec1[b][1] + img0Col

                slope= (x1-x0)/(y1-y0)

                for d in range(y1-y0):
                    comp[max(min(round(x0+slope*d), max(img0Row,img1Row)),0), y0+d, 0] = 1
    s = "Found " + str(count) + " stuff"
    print(count)
    eucList.sort()
    print(eucList[:50])
    print(max(eucList))
    plt.imshow(comp)
    plt.show()


# scale-space extrema detection
def sift(img, sigma = 0.5, numInterval = 3):
    imgGray = grayScale(img)
    imgs = gaussPyramid(imgGray, 4, numInterval, sigma)
    imgsDiff = diffGaussPyramid(imgs)
    candidates = candidatePyramid(imgsDiff)
    print("found candidates")
    keypoints = findPyramidKeypoints(candidates, imgsDiff, 2.5, 3)
    orientedKeys = assignPryaOri(keypoints, imgs, numInterval, sigma)
    orientedKeys = removeDuplicates(orientedKeys)
    print("oriented keys")
    descript = generateDescriptors(orientedKeys, imgs)
    print("added descriptors")

    return descript

# realign the scaled keypoints back to the original dimensions and also associate a descriptor
def realignKP(keypoints, descriptors):
    aligned = []
    for i in range(len(keypoints)):
        octave = keypoints[i][3]
        x = keypoints[i][0] * 2**octave
        y = keypoints[i][1] * 2**octave
        aligned.append((x,y,descriptors[i]))
    
    return aligned


# generates descriptors from keypoints 
def generateDescriptors(keypoints, pyramid, winWidth=16, thres = 1000):
    # keypoints in [scaled x, scaled y, scale, octave #, orientation] format
    descriptors = []
    sum = len(keypoints)
    counter = 0
    for kp in keypoints:
        # create a windowWidth/4 x windowWidth/4 x 8 vector to store the descriptors (8=number of degree bins)
        # adding 2 to the number of windowWidth bins to account for boundary bins that will be removed later
        # bins are divided into -180~-136, -135~-91,...,270~314,315~360 degrees
        descriptiveVec = np.zeros(shape=(int(winWidth/4),int(winWidth/4),8))

        # create a windowWidth * windowWidth to calculate the magnitude and orientation for section of the sample patch
        sampleWindow = np.empty(shape=(winWidth+1, winWidth+1), dtype = object)
        smoothedSamples = np.zeros(shape = (winWidth+2, winWidth+2, 8))

        # extract the position, scale, octave and orientation of the point 
        xP = kp[0]
        yP = kp[1]
        zP = kp[2]
        octave = kp[3]
        angle = kp[4]
        cosine = math.cos(math.radians(angle))
        sine = math.sin(math.radians(angle))

        counter += 1
        ## IMPORTANT
        # print(counter/sum)

        # extract the gaussian where the keypoint comes from
        octi = pyramid[octave]

        scale = octi[zP]
        imgRow, imgCol = scale.shape
    
        # weight the gaussian sigma as half the windowwidth
        halfWidth = int(winWidth/2)
        weightFac = -1/2 * 1/halfWidth**2

        # for every "point" record the partial magnitude and orientation to...
        # ...later be sorted into different bins
        for x in range(-halfWidth, halfWidth+1):
            for y in range(-halfWidth, halfWidth+1):
                xR = round(x * cosine - y * sine + xP)
                yR = round(x * sine + y * cosine + yP)
                
                # only record if the rotated coordinates are within bounds
                # also account for the later magnitude and orientation calculations
                if xR < 1 or xR > imgRow-2 or yR < 1 or yR > imgCol-2:
                    sampleWindow[x+halfWidth, y+halfWidth] = (-1, 0, 0, 0)
                else:
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
                    sampleWindow[x+halfWidth, y+halfWidth] = [gMag, binOri, xBin, yBin]
        
        # cycle through every sample pixel and use inverse trilinear interpolation to...
        # ...redistribute the magnitudes
        for i in range(winWidth+1):
            for j in range(winWidth+1):
                gMag = sampleWindow[i,j][0]
                binOri = sampleWindow[i,j][1]
                xBin = sampleWindow[i,j][2]
                yBin = sampleWindow[i,j][3]

                # make sure the sample is valid
                if gMag > 0:
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

                    smoothedSamples[i,j,binFloor % 8] += c000
                    smoothedSamples[i,j,(binFloor+1) % 8] += c001
                    smoothedSamples[i,j+1,binFloor % 8] += c010
                    smoothedSamples[i,j+1,(binFloor+1) % 8] += c011
                    smoothedSamples[i+1,j,binFloor % 8] += c100
                    smoothedSamples[i+1,j,(binFloor+1) % 8] += c101
                    smoothedSamples[i+1,j+1,binFloor % 8] += c110
                    smoothedSamples[i+1,j+1,(binFloor+1) % 8] += c111

        # sum up the smoothed samples into a more concentrated 4x4x8 array
        for a in range(int(winWidth/4)):
            for b in range(int(winWidth/4)):
                for c in range(8):
                    descriptiveVec[a,b,c]  +=  smoothedSamples[a*4+1,b*4+1,c] + \
                                            smoothedSamples[a*4+2,b*4+1,c] + \
                                            smoothedSamples[a*4+3,b*4+1,c] + \
                                            smoothedSamples[a*4+4,b*4+1,c] + \
                                            smoothedSamples[a*4+1,b*4+2,c] + \
                                            smoothedSamples[a*4+2,b*4+2,c] + \
                                            smoothedSamples[a*4+3,b*4+2,c] + \
                                            smoothedSamples[a*4+4,b*4+2,c] + \
                                            smoothedSamples[a*4+1,b*4+3,c] + \
                                            smoothedSamples[a*4+2,b*4+3,c] + \
                                            smoothedSamples[a*4+3,b*4+3,c] + \
                                            smoothedSamples[a*4+4,b*4+3,c] + \
                                            smoothedSamples[a*4+1,b*4+4,c] + \
                                            smoothedSamples[a*4+2,b*4+4,c] + \
                                            smoothedSamples[a*4+3,b*4+4,c] + \
                                            smoothedSamples[a*4+4,b*4+4,c]
                    # adjust vector to threshold if larger than that
                    if thres < descriptiveVec[a,b,c]:
                        descriptiveVec[a,b,c] = thres
                    
        descriptiveVec = descriptiveVec.flatten()
        
        # if the descriptor is positive, realign the coordinates to max scale and record descriptor
        if np.max(descriptiveVec) > 0:
            descriptiveVec /= np.max(descriptiveVec)
            descriptors.append((xP * 2**octave,yP * 2**octave,descriptiveVec))

    return descriptors


# remove duplicate keypoints
def removeDuplicates(keypoints):
    # sorting multiple attribute arrays: https://stackoverflow.com/questions/31942169/python-sort-array-of-arrays-by-multiple-conditions
    sortKP = lambda x:(x[0], x[1], x[2], x[3], x[4])
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
            lastKP[4] != nextKP[4]:
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
                        if yP + y > 0 and yP + y < imgCol-1:
                            currX = xP + x
                            currY = yP + y
                            # assign a gradient magnitude and orientation (converting from -pi to pi radians to 0-360 degrees for bin) according to the paper
                            gMag = math.sqrt((scale[currX+1,currY] - scale[currX-1,currY])**2 + (scale[currX, currY+1] - scale[currX, currY-1])**2)
                            ori = np.arctan2(scale[currX,currY+1] - scale[currX, currY-1], scale[currX+1,currY] - scale[currX-1,currY])/math.pi * 180 + 180

                            # the gaussian weight, the constant doesn't really matter because its relative
                            weight = math.exp(-(x**2 + y**2)/(2*currSD**2)) 

                            # get the bin index 
                            bin = math.floor(ori/10)

                            # add the weighted magnitude to the bin
                            histo[bin] += weight * gMag
        
            # record the maximum bin peak orientation, and any other orientations that are within 80% of the max peak
            limPeak = max(histo) * 0.8
            for k in range(len(histo)):
                if histo[k] >= limPeak:
                    # interpolate the orientation parabolically and add to the array of keypoints
                    # [scaled x, scaled y, scale, octave #, orientation]
                    oriKeys.append((xP,yP,zP, i, fitParabola(k, histo)))

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
    return  interloIndex * 10 - 180
            

# returns qualified keypoints that are localized, not low contrast, and not on an edge
def findPyramidKeypoints(candidates, pyramid, edgeThres = 2.5, eigenRatio = 3):
    pyraKeys = []
    for i in range(len(pyramid)):
        pyraKeys.append(findOctaveKeypoints(candidates[i], pyramid[i], edgeThres, eigenRatio))

    return pyraKeys

# returns the qualified keypoints of an octave by
# offsetting it, checking for low contrast, and whether it's on the edge

# heavy guidance from the original paper and https://dsp.stackexchange.com/questions/10403/sift-taylor-expansion
# for additional explanations
def findOctaveKeypoints(octCandidates, oct, edgeThres, eigenRatio):
    keypoints = []

    # cycle through the candidates of the octave
    conv = 0
    edged = 0
    final = 0
    for cand in octCandidates:
        x = cand[0]
        y = cand[1]
        z = cand[2]

        imgRow, imgCol = oct[0].shape 

        outofBounds = False
        converged = False

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
                converged = True
                break
            
            # update the keypoint position
            x += int(round(offset[0]))
            y += int(round(offset[1]))
            z += int(round(offset[2]))

            # if the offset is out of bounds, skip it
            if x < 1 or x >= imgRow-1 or y < 1 or y >= imgCol-1 or z < 1 or z >= len(oct)-1:
                outofBounds = True
                break
        
        # if the offset isn't out of bounds continue to further checks AND
        # ...if the point converged within the maximum number of iterations
        if not outofBounds and converged:
            # get the function value at the extrenum and check for low contrast
            funcV = oct[z][x,y] + np.dot(grad, offset)/2
            conv += 1
            
            # if the function value is greater than or equal to 0.03, check to see if point is on edge

            # paper correct thres = 0.03
            # numbers-wise correct thres = 2.5
            # visually correct = 5.0
            if abs(funcV) >= edgeThres:
                edged += 1
                # calculate the trace and determinant based on 2x2 Hessian
                trace = dxx + dyy
                det = dxx*dyy - dxy**2

                # if the determinant is positive, continue (non-positive means det is point is not extrenum
                # as the curvature has different sides)
                if det > 0:

                    # check the ratio principal curvature to see if it is below a threshold (not an edge)

                    # paper correct thres = 10
                    # numbers-wise correct threshold = 3
                    if trace**2/det < (eigenRatio+1)**2/eigenRatio:
                        final += 1
                        # add keypoint if it has cleared all conditions
                        keypoints.append([x,y,z])

    # Number of candidates for testing purposes
    # print("Oct candidates " + str(len(octCandidates)))
    # print("Converged candidates " + str(conv))
    # print("Edged candidates " + str(edged))
    # print("Final candidates " + str(final))
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
                if (currMx == oct[i][j,k] and max(aboveMx, belowMx) < oct[i][j,k]) or \
                    (currMn == oct[i][j,k] and min(aboveMn, belowMn) > oct[i][j,k]):
                    
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
    if len(sys.argv) < 4:
        pic0 = os.getcwd() + "\\" + sys.argv[1]
        pic1 = os.getcwd() + "\\" + sys.argv[2]
        img0 = mpimg.imread(pic0)
        img1 = mpimg.imread(pic1)
        showSpots(img0, sift(img0))

        compareTwo(img0, img1, sift(img0), sift(img1))
    else:
        print("Function requires only 1 picture")
        raise SystemExit