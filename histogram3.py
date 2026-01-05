import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread("aks/clahe_2.jpg")
image2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#کنتراستشو میبرم بالا بعد هیستوگرامشو نشون میدم
equalized_img = cv.equalizeHist(image2)
histogram2 = cv.calcHist([equalized_img], [0], None, [256], [0, 256])
histogram1 = cv.calcHist([image2], [0], None, [256], [0, 256])


clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(image2)
histogram3 = cv.calcHist([cl1], [0], None, [256], [0, 256])


plt.subplot(322)
plt.plot(histogram1)
plt.title('Cmap: Histogram1')

plt.subplot(321)
plt.imshow(image2, 'gray')
plt.title('Cmap: image2')

plt.subplot(323)
plt.imshow(equalized_img, 'gray')
plt.title('Cmap: Equalized Image')

plt.subplot(324)
plt.plot(histogram2)
plt.title('Cmap: Histogram2')

plt.subplot(325)
plt.imshow(cl1, 'gray')
plt.title('Cmap: cl1')

plt.subplot(326)
plt.plot(histogram3)
plt.title('Cmap: Histogram3')

plt.show()
