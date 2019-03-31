import cv2
import sys
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
def ojas(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
                cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)	
                for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                        cap = cv2.VideoCapture(cv2.CAP_ANY)

        # Display the resulting frame
        while True:
                ret, img = cap.read()
                if ret == True:
                        cv2.imshow('Live',ojas(image)) 
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

cap.release()
cv2.destroyAllWindows()  
