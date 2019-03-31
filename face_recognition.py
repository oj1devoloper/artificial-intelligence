import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer1.yml')
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

ID = 0
Names = ['none', 'Ojasvi', 'bindu', 'yug']


cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4, 480)

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

#cv2.putText(im, str(Id), (x,y+h), fontface, fontscale, fontcolor) 

#cv2.putText(im,"Cat",(x,y-10),font,0.55,(0,255,0),1)
#def getId(ID):

while True:
    ret, im = cam.read()
    img = cv2.flip(im, -1)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(
    gray,
    scaleFactor = 1.2,
    minNeighbors = 5,
    minSize = (int(minW), int(minH)),
    )
    
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        ID, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<100):
            ID = Names[ID]
            conf = " {0}%".format(round(100 - conf))
        else:
            ID = "unknown"
            conf = " {0}%".format(round(100 - conf))

            cv2.putText(im, str(ID), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(im, str(conf), (x+5,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 1)        

        #cv2.rectangle(im, (x-22,y-40), (x+w+22, y-40), (0,255,0), -1)
        #cv2.putText(im, str(Id), (x,y-40), fontface, fontscale, fontcolor, 2)
        #cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 2)
        cv2.putText(im,"Ojasvi",(x,y-10),0.55,(0,255,0),1)
        cv2.imshow('Frame', im) 
        if cv2.waitKey(10):
            break

cam.release()
cv2.destroyAllWindows()
