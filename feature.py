import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread("aks/feature.jpg")
image2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)

#فیچرای عکسو میخوام پیدا کنم


#sift
feature_sift = cv.SIFT_create()
sift_keypoints , descriptor1 = feature_sift.detectAndCompute(image2, None)
image_sift = cv.drawKeypoints(image2,sift_keypoints,None)
#surf

#orb(best)
feature_orb = cv.ORB_create(nfeatures=1500)
orb_keypoints , descriptor3 = feature_orb.detectAndCompute(image2, None)
image_orb = cv.drawKeypoints(image2,orb_keypoints,None)

# تو این خط دوتا عکسو کنار هم میذاریم
plt.figure(figsize = (10,10))

#اینجا هم نمایش میدیم
plt.subplot(1,2,1)
plt.imshow(image_orb)
plt.title('Cmap: ORB')

plt.subplot(1,2,2)
plt.imshow(image_sift)
plt.title('Cmap: SIFT')
plt.show()
