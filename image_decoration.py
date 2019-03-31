import numpy as np
import cv2
from matplotlib import pyplot as plt

BLUE = (255,0,0)

img = cv2.imread("opencv-logo.png")

replicate = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT)

plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGNAL')
plt.subplot(232),plt.imshow(img,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(img,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(img,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(img,'gray'),plt.title('CONSTANT')

plt.show()