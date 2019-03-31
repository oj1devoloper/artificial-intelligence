import cv2
from PIL import Image
import numpy as np
import time
cascPath = "haarcascade_frontface_default.xml"
maskPath = "funny.png"

faceCascade = cv2.CascadeClassifier(cascPath)
mask = Image.open(maskPath)
def fun_mask(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 		
	faces = faceCascade.detectMultiScale(gray ,1.15)

	background = Image.fromarray(image)

	for (x,y,w,h) in faces:
		resized_mask = mask.resize((w,h), Image.ANTIALIAS)
		offset = (x,y)
		background.paste(resized_mask, offset, mask=resized_mask)
	return np.asarray(background)
cap = cv2.VideoCapture(cv2.CAP_ANY)

while True:
	ret,frame = cap.read()
	if ret == True:
		cv2.imshow('Live', fun_mask(frame))
		if cv2.waitKey(1) == 27 & 0XFF == ord('q'):
cap.release()
cv2.destroyAllWindows()
