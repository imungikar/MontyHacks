"""
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import math
import time

from cvzone.HandTrackingModule import HandDetector


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
#class_names = ["A","B","C","D","E"]


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
    


def getASL(img):



    img_copy = img.copy()
    hands, img = detector.findHands(img)
    img = img-img_copy
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgBlack = np.zeros((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgBlack[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgBlack[hGap:hCal + hGap, :] = imgResize

        
        cv2.imshow("ImageBlack", imgBlack)
        
        # Resize the raw image into (224-height,224-width) pixels
        imgBlack= cv2.resize(imgBlack, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Show the image in a window
        #cv2.imshow("Webcam Image", img)

        # Make the image a numpy array and reshape it to the models input shape.
        imgBlack = np.asarray(imgBlack, dtype=np.float32).reshape(1, 224, 224, 3)
        
        # Normalize the image array
        imgBlack = (imgBlack / 127.5) - 1

        # Predicts the model
        prediction = model.predict(imgBlack)
        
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name, end="\n")
        print("Confidence Score:", str(
            np.round(confidence_score * 100))[:-2], "%")
        

    return img
"""
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
 
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
 
offset = 20
imgSize = 300
 
folder = "Data/C"
counter = 0
 
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",]
 
def getASL(img):
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
 
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
 
        imgCropShape = imgCrop.shape
 
        aspectRatio = h / w
 
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
 
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
 
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)
 
 
    return imgOutput