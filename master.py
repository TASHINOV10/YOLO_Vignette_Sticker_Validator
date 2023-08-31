
#Install necessary libraries
import threading
import queue
import torch
import cv2
import pytesseract
import pandas as pd
import re
import time
from datetime import datetime
import json
import requests
import os

#Set the paths according to your setup
tesseract_executable = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_executable
weightsPath = 'C:/Users/Iliyan Tashinov/Desktop/YOLOv5_Vignette_Sticker_Validator/weights/epochs_100/last.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path= weightsPath, force_reload=True)
video = 'C:/Users/Iliyan Tashinov/Desktop/YOLOv5_Vignette_Sticker_Validator/video_test.mp4'

#Create a directory for storing the extracted license plates
if not os.path.exists("crops"):
    os.makedirs("crops")

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


def extract_license(crop_path):
    crop_contoured = cv2.imread(crop_path,cv2.COLOR_BGR2GRAY)
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
    data = detections
    crop = get_crop(gray_frame, data)
    crop_contoured = get_contour1(crop)
    global filePath
    filePath = f'C:/Users/Iliyan Tashinov/Desktop/YOLOv5_Vignette_Sticker_Validator/crops/crop_contoured{frame_count}_{int(ith_detection)+1}.png'
    cv2.imwrite(filePath, crop_contoured)


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
        validation_lst.append(result_text)
        time_remaining_lst.append(time_left)
        numPlt_lst.append(license_plate)
        numPlt_lst_1.append(license_plate)
        time_remaining_lst_1.append(time_left)
        date_lst.append(datetime.today().strftime("%d/%m/%Y %H:%M:%S"))
        return 0

    result_text = "Vignette is valid"
    time_left = p['vignette']['validityDateToFormated']
    numPlt_lst_1.append(license_plate)
    time_remaining_lst_1.append(time_left)
    validation_lst.append(result_text)
    time_remaining_lst.append(time_left)
    numPlt_lst.append(license_plate)
    date_lst.append(datetime.today().strftime("%d/%m/%Y %H:%M:%S"))
    return 0

#set empty lists
crop_index = []
numPlt_lst = []
date_lst =[]
validation_lst = []
time_remaining_lst = []
time_lst = []
crop_que= queue.Queue()
extracted_que = queue.Queue()
numPlt_lst_1 = []
time_remaining_lst_1 = []

start_point = (0, 0)
end_point = (380, 100)
color = (50, 50, 50)
thickness = -1

print("everything loaded up until here")
def printPath(queue):
    crop_path = queue.get()
    license = extract_license(crop_path)
    plate_format = re.compile('^[a-zA-Z]{2}[0-9]{4}[a-zA-Z]{2}$')
    plate_format1 = re.compile('^[a-zA-Z]{1}[0-9]{4}[a-zA-Z]{2}$')

    if (plate_format.match(license) is not None or plate_format1.match(license) is not None) and license not in numPlt_lst:
        extracted_que.put(license)
        check_license(license)
        return 0
    return 0

def printConsoleOutput(frame):
    cv2.rectangle(frame, start_point, end_point, color, thickness)

    cv2.putText(frame, "CONSOLE", (50, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 180, 0), 1)

    x, y = 20, 50  # Position of the label in the window
    line_height = 18  # Height between each list item
    for idx, item in enumerate(numPlt_lst_1):
        cv2.putText(frame, item, (x, y + idx * line_height), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,180,0), 1)

    x, y = 115, 50  # Position of the label in the window
    line_height = 18  # Height between each list item
    for idx, item in enumerate(time_remaining_lst_1):
        cv2.putText(frame, item, (x, y + idx * line_height), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,180,0), 1)

    if len(numPlt_lst_1) > 2:
            numPlt_lst_1.pop(0)
            time_remaining_lst_1.pop(0)



# load video
cap = cv2.VideoCapture(video)
frame_count = 1

# start video
while cap.isOpened():


    start_time = time.time()
    ret, frame = cap.read()  # ret is boolean, returns true if frame is available;
    frame_resized = cv2.resize(frame, (608, 608))  # resizing the frames for better utilization of YOLO
    results = model(frame_resized)  # run YOLO on the resized frame

    printConsoleOutput(frame_resized)

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
        crop_que.put(filePath)
        t1 = threading.Thread(target=printPath,args = (crop_que,))
        t1.start()

    time_lst.append(time.time() - start_time)

    frame_count += 1

    # display frame
    cv2.imshow('YOLO', frame_resized)

    if cv2.waitKey(10) & 0xFF == ord('s') or frame_count == 100:
        break

cap.release()
cv2.destroyAllWindows()

t1.join()

print("waiting before print")
for i in range(0,11):
    time.sleep(1)
    print(f'{i} seconds')

data = list(zip(numPlt_lst,date_lst,validation_lst, time_remaining_lst))
df = pd.DataFrame(data, columns=["number plate","detected on", "vignette status", "date-time of expiry"])
print("-------------------------------------Data Export------------------------------------------")
print(df)

def Average(lst):
    return sum(lst) / len(lst)

average = Average(time_lst)
print("-------------------------------------Performance Stats-------------------------------------")
print("Average time needed for a frame to load=", round(average, 2))
print(f"Total Frames = {frame_count}")
