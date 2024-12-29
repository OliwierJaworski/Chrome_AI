import time
import os
from ultralytics import YOLO
import torch
# Capture
import mss
from PIL import Image, ImageGrab
import pyautogui

import cv2
import numpy as np
import math

pretrained_path = "C:/DERDEJAAR/SmartSystem/githubown/Chrome_AI/src/results/model/default_model/custom_model_acht.pt" #which model will be used if pretrained is True
mode_save_name = "custom_model_acht.pt" #where the model will be saved 

path = "C:/DERDEJAAR/SmartSystem/githubown/Chrome_AI/gameplay/gamePlay.mp4"
src = cv2.imread(path)

def model_load():
    model = YOLO(model=pretrained_path)
    return model

def model_test_frame(model, frame):
    # Convert the frame to the format expected by the YOLO model
    frame = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # YOLO expects RGB images
    results = model(frame)  # Pass the frame directly to the model
    return results

def calculate_distance(box1, box2):
    """
    Calculate the Euclidean distance between two bounding box centroids.
    :param box1: [x1, y1, x2, y2] for the first box
    :param box2: [x1, y1, x2, y2] for the second box
    :return: distance between the centroids
    """
    centroid1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    centroid2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
    return distance

try:
    
    model = model_load()
    img = None

    cap = cv2.VideoCapture("C:/DERDEJAAR/SmartSystem/githubown/Chrome_AI/gameplay/gamePlay.mp4")
    ret, frame = cap.read()
    while(1):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)

        results = model_test_frame(model, img)
        detected_boxes = results[0].boxes.xyxy.cpu().numpy()

        if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
            cap.release()
            cv2.destroyAllWindows()
            break

        cv2.imshow('frame',frame)

    
except Exception as e:
    # Catch and print any errors that occur
    print(f"Debug: An error occurred: {e}")


   