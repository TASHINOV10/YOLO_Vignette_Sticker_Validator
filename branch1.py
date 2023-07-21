import threading
import time

#-------------MAIN THREAD---------------------
#YOLO recognizes the license plate in the frame
#crops the area around the license plate
#####saves the crop in a directory------------------------> [START THREAD]
#display the frame and the results from the extraction if any
#-----------------END-------------------------


#--------------------------THREAD---------------------
#run OCR on crop
#check the extracted license if it fits format
#return result to main thread
#------------------------ACTIVATE AGAIN IF CROP IS THERE-----------------

'''
def extractLicense(frames):

    while True:
        global stop_threads
        time.sleep(1)
        print(f"thread works at frame {frames}")
        if stop_threads:
            break


licensePlate = ''
count = 0
global frames
frames = 10
it = 4



#running the video
while frames != 0:
    stop_threads = False
    print(stop_threads)
    t1 = threading.Thread(target=extractLicense, args=(frames,))
    t1.start()



    print(f'Main Thread - frame: {-1 * (frames - 10)}')
    licensePlate = "license plate: " + str(count)

    time.sleep(4)

    frames -= 1
    count +=1
    stop_threads = True

t1.join()
'''
#------------------------------------------MAIN CODE--------------------------------------

#Install necessary libraries

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

#Set the paths according to your setup
tesseract_executable = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_executable
weightsPath = 'C:/Users/Iliyan Tashinov/Desktop/YOLO_Vignette_Sticker_Validator/weights/epochs_100/last.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path= weightsPath, force_reload=True)
video = 'C:/Users/Iliyan Tashinov/Desktop/YOLO_Vignette_Sticker_Validator/video_test.mp4'

#Define the functions

def format_frame(frame):
    resultimage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return resultimage

def draw_rectangle(frame, detection):
    data = results.xyxy[0][i]
    a = (int(data[0]), int(data[1]))
    b = (int(data[2]), int(data[3]))

    cv2.rectangle(frame, a, b, (0, 255, 0), 3)
    return a, b

def get_frameProportions(frame):
    a, b = frame.shape

    a_p = a / 608
    b_p = b / 608

    return a_p, b_p


def get_crop(frame, detections):
    a, b = get_frameProportions(frame)
    data = detections
    crop = frame[int(detections[1] * a):int(detections[3] * a),
           int(detections[0] * b):int(detections[2] * b)]
    return crop


def get_contour1(crop):
    threshold = cv2.threshold(crop, 85, 255, cv2.THRESH_BINARY)[1]
    crop_contoured = cv2.bitwise_not(threshold)

    return crop_contoured


def extract_license(crop_contoured):

    analysis = cv2.connectedComponentsWithStats(crop_contoured,4,cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    symbolList = []
    xCentroidList = []

    for i in range(1, totalLabels):

        area = values[i, cv2.CC_STAT_AREA]
        X, Y = centroid[i]

        if (area < 50) or (area > 1000):
            continue

        x1 = values[i, cv2.CC_STAT_LEFT]
        y1 = values[i, cv2.CC_STAT_TOP]
        w = values[i, cv2.CC_STAT_WIDTH]
        h = values[i, cv2.CC_STAT_HEIGHT]

        # convert it to 255 value to mark it white
        component = (label_ids == i).astype("uint8") * 255

        # Increase brightness by multiplying with a scaling factor
        brightness_scale = 2  # Adjust this value to increase or decrease brightness
        component = component * brightness_scale
        cropped_component = component[y1 - 6:y1 + h + 6, x1 - 6:x1 + w + 6]

        try:
            symbol = pytesseract.image_to_string(cropped_component,
            config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3')
            symbol = symbol.replace('\n', '')
            print(symbol)
            symbolList.append(symbol)
            xCentroidList.append(X)

        except:
            continue

    print(symbolList)
    xCentroidList, symbolList = zip(*sorted(zip(xCentroidList, symbolList)))
    license_plate = ''.join(symbolList)

    return license_plate


def get_extraction(gray_frame, detections, frame_count,ith_detection):
    frame_count = frame_count
    plate_format = re.compile('^[a-zA-Z]{2}[0-9]{4}[a-zA-Z]{2}$')
    plate_format1 = re.compile('^[a-zA-Z]{1}[0-9]{4}[a-zA-Z]{2}$')
    data = detections
    crop = get_crop(gray_frame, data)
    crop_contoured = get_contour1(crop)
    cv2.imwrite(f'C:/Users/Iliyan Tashinov/Desktop/YOLO_Vignette_Sticker_Validator/crops/crop_contoured{frame_count}_{int(ith_detection)+1}.png', crop_contoured)

'''
    license_plate = extract_license(crop_contoured)
    print(license_plate)

    if plate_format.match(license_plate) is not None or plate_format1.match(license_plate) is not None:
        return crop_contoured, license_plate

    license_plate = "N/A"
    return crop_contoured, license_plate
'''

def check_license(license_plate):
    my_url = 'https://check.bgtoll.bg/check/vignette/plate/BG/' + license_plate
    x = requests.get(my_url)
    if (x.status_code != 200):
        result_text = "error"
        time_left = "error"
        return result_text, time_left

    p = json.loads(x.content)

    if (p['vignette'] is None):
        result_text = "Vignette not valid."
        time_left = "N/A"
        return result_text, time_left

    result_text = "Vignette is valid"
    time_left = p['vignette']['validityDateToFormated']
    return result_text, time_left


#set empty lists
crop_index = []
number_plate = []
date =[]
validation_df = []
time_remaining_lst = []

# load video
cap = cv2.VideoCapture(video)
frame_count = 1

# start video
while cap.isOpened():
    ret, frame = cap.read()  # ret is boolean, returns true if frame is available;
    frame_resized = cv2.resize(frame, (608, 608))  # resizing the frames for better utilization of YOLO
    results = model(frame_resized)  # run YOLO on the resized frame
    print(f'-----number of plates detected = {len((results.xyxy[0]))} - at frame {frame_count}--------------')
    #    ---      ---      ---        0  1  2  3  4  5
    # loop through elements in tensor[h1,w1,h2,w2,cf,lb]
    for i in range(len(results.xyxy[0])):  # len represents the number of tensors;n tensors is the number of detections;#the loop is executed for each tenso

        print(f'checking confidence level of detection - {i + 1} ')

        conf = results.xyxy[0][i][4]  # take the ith tensor/detection ; #conf = data[4] #take the confidence of that detection

        if (conf < 0.80):  # checking how confident is the model; not acceptable under 80%
            print("detection's confidence is too low")
            print(f'++++end of analysis for detection number++++ {i + 1}')
            continue

        print(f'passed the confidence test')

        a, b = draw_rectangle(frame_resized, results.xyxy[0][i])
        gray_frame = format_frame(frame)
        extraction = get_extraction(gray_frame, results.xyxy[0][i], frame_count,i)

    frame_count += 1

    # display frame
    cv2.imshow('YOLO', frame_resized)

    if cv2.waitKey(10) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()


