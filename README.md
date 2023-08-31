# YOLOv5_Vignette_Sticker_Validator

This repository contains code for license plate recognition using the YOLOv5 (You Only Look Once) object detection model. The code allows you to detect vehicle license plates in videos and then validate whether the vehicle in question has valid vignette sticker. Validation requests are done through the website of the Bulgarian National Toll Administraiton.

![gif4e](ezgif.com-video-to-gif.gif)

## Getting Started

To get started with the code, follow the steps below:

### Clone the Repository

Clone this GitHub repository by running the following command:

```python
git clone https://github.com/TASHINOV10/YOLOv5-Vignette-Sticker-Validator
```

### Install Dependencies

Navigate to the cloned repository and install the required libraries by running the following command:

```python
pip install -r requirements.txt
```
### Install Necessary Libraries

Import the necessary libraries in your Python script:

```python
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
```

### Set the paths according to your setup

- Before running the OCR model, you need to define the path to the Tesseract executable file. The executable file is required for the OCR functionality to work properly. Follow the steps below to set the Tesseract path:

1. After running the requirements file , locate the directory where Tesseract is installed on your system. By default, it is usually installed in "C:\Program Files\Tesseract-OCR\".
2. Proceed  by running the below code. Make sure to do it every time when running the program.
   
```python
tesseract_executable = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_executable
```

- Use the weights provided in the already cloned repository to run the trained YOLO model.
  
```python
weightsPath = '.../epochs_100/last.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path= weightsPath, force_reload=True)
```

- You can test the program on any video file. For simplicity you can use the already provided test video in the cloned repository.
  
```python
video = '.../video_test.mp4'
```
## Process Flow Diagram

![diagram](diagram.jpeg)
