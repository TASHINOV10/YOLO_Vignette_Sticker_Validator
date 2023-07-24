import time
import threading
import os
import queue
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pytesseract
import os
import pandas as pd
import re
from PIL import Image
from re import compile
import time
from datetime import datetime
import json
import requests



my_list = [1,2,3,4,5,6,7,8,10]
print(my_list[-1])
print(my_list[-2])
print(my_list[-3])

def draw_label(frame, label_list):
    x, y = 20, 50  # Position of the label in the window
    line_height = 30  # Height between each list item
    for idx, item in enumerate(label_list):
        cv2.putText(frame, label_list.pop(), (x, y + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Replace 'your_video.mp4' with the path to your video file or 0 for webcam
cap = cv2.VideoCapture('C:/Users/Iliyan Tashinov/Desktop/YOLO_Vignette_Sticker_Validator/video_test.mp4')
frame_count = 0
while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()
    my_list.append(str(frame_count))
    frame_resized = cv2.resize(frame, (1000, 700))
    if not ret:
        break
    time.sleep(1)
    # Here, you can perform any image processing or video analysis you want on 'frame'

    # Display the list as a label on the frame
    draw_label(frame_resized, my_list)

    # Show the frame with the label
    cv2.imshow('Video with List Label', frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
