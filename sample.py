import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
import random
import time


# Webcam frame
cap = cv2.VideoCapture(0)
cap.set(3, 1280)		#width
cap.set(4, 720)			#height


#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
 success, img=cap.read()
 # img = cv2.flip(img ,1)
 hands, img = detector.findHands(img)
 if hands:
  lmList = hands[0]['lmList']
  print(lmList)
 cv2.imshow("Image",img)
 key =cv2.waitKey(1)