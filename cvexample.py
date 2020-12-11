

import numpy as np
import sys
import math
import os
import cv2
import matplotlib.pyplot as plt

pic0 = os.getcwd() + "\\" + sys.argv[1]
#reading image
img1 = cv2.imread(pic0)  
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#keypoints
sift = cv2.SIFT_create()
kp = sift.detect(gray1,None)

img_1 = cv2.drawKeypoints(gray1,kp,img1)
# plt.imshow(img_1)
# cv2.waitKey(0)
# plt.show()
cv2.imwrite('test\\sift_keypoints.jpg',img_1)