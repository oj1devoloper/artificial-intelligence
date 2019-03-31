import cv2
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3, 640) # set video width
cap.set(4, 480) # set video height
face = cv2.CascadeClassifier("C:\\Users\\ojasv\\Desktop\\practice\\haarcascade_frontalface_default.xml")
ID = input("enter your id: ")
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
sampleNum = 0
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        sampleNum = sampleNum + 1
        cv2.imwrite('DataSet/user.' +str(ID)+ '.' +str(sampleNum)+ '.jpg' ,gray[y:y+h,x:x+w])
        cv2.imshow('Frame', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif sampleNum >= 30: # Take 30 face sample and stop video
            break
        print("[INFO] saving the images and cleening up..")
cap.realease()
v2.destroyAllWindows()
        
