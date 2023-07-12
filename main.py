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


tesseract_executable = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_executable
weightsPath = 'C:/Users/Iliyan Tashinov/Desktop/yolo_project/weights/epochs_100/last.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path= weightsPath, force_reload=True)
video = 'C:/Users/Iliyan Tashinov/Desktop/YOLO_Vignette_Sticker_Validator/video_test.mp4'



print("test")