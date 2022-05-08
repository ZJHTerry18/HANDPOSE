import cv2
import numpy as np
import os

VIDEO_PATH = "videos/test2.mp4"
SAVE_PATH = "images/test2/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


def save_image(image,addr,num):
  address = addr + str(num).zfill(4) + '.jpg'
  cv2.imwrite(address,image)
  
videoCapture = cv2.VideoCapture(VIDEO_PATH)


success, frame = videoCapture.read()
i = 0
timeF = 1
j=0
while success :
  i = i + 1
  if (i % timeF == 0):
    j = j + 1
    save_image(frame,SAVE_PATH,i)
  success, frame = videoCapture.read()
