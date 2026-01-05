import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread("aks/sc2.jpg")
image2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
width ,height ,channels = image2.shape
mask = np.zeros(image2.shape[0:2], np.uint8)
mask[0:int(width/2), 0:int(height/2)] = 255

color = ['r' , 'g' , 'b']

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
for i,col in enumerate(color):
  histogram = cv.calcHist([image2], [i], None, [256], [0, 256])
  plt.plot(histogram, color=col)
  plt.title('Cmap: Histogram')

plt.subplot(1,2,2)
plt.imshow(image2)
plt.title('Cmap: image2')
plt.show()
