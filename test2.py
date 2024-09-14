import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3
import speech_recognition as sr

# Initialize the text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

class_images = {
    "A": cv2.imread("images/A.jpg"),
    "B": cv2.imread("images/B.jpg"),
    "C": cv2.imread("images/c.jpg"),
}

offset = 20
imgSize = 300
labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
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
            prediction, index = classifier.getPrediction(imgWhite)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        if labels[index] in class_images:
            output_img = class_images[labels[index]]
            # Resize the output_img to fit the output window
            output_img = cv2.resize(output_img, (img.shape[1], img.shape[0]))           
            cv2.imshow("Output", output_img)
              
             # Read out the label using text-to-speech
            engine.say(labels[index])
            engine.runAndWait()
            
        else:
            print(f"No image defined for class {labels[index]}")

        #cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)
