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
