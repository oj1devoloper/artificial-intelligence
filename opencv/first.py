import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('images.jpg',0)
blur = cv2.blur(img,(10,10))
plt.subplot(122),plt.imshow(img),plt.title("Orignal")
plt.subplot(121),plt.imshow(blur),plt.title("blurred")
plt.show() 