
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread("aks/Xray_share.jpg")
image2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
histogram = cv.calcHist([image2], [0], None, [256], [0, 256])

# ایجاد یک شکل (Figure) با دو بخش کنار هم
plt.figure(figsize=(12, 6))


# تصویر دوم در سمت راست
plt.subplot(1, 2, 2)
plt.imshow(image2,cmap='gray')
plt.title('Cmap: Gray')

#تصویر سوم در سمت چپ
plt.subplot(1,2,1)
plt.plot(histogram)
plt.title('Cmap: Histogram')

plt.show()
