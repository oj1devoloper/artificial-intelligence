import cv2,os
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
path = "DataSet"
def getImagesAndLabels(path):
    # get the path of all files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create an empty face list
    faceList = []
    # creates an empty id list
    ids = []
    for imagePath in imagePaths:
        pilimage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilimage, 'uint8') 
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=face.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
             faceList.append(imageNp[y:y+h,x:x+w])
             ids.append(ID)
        return faceList,ids

faces,ids = getImagesAndLabels(path)
recognizer.train(faces,np.array(ids))
recognizer.save('trainer/trainer1.yml')
    
