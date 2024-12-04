# pip install selenium
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import os
from ultralytics import YOLO
import torch
# Capture
import mss
from PIL import Image, ImageGrab
import pyautogui

import cv2 as cv
import numpy as np
import math

# Path to your ChromeDriver
CHROME_DRIVER_PATH = 'C:\\Program Files\\chrome-win64\\chrome-win64\\chrome.exe'

# Start WebDriver
service = Service(CHROME_DRIVER_PATH)
options = webdriver.ChromeOptions()
#options.add_argument('--headless')
#options.add_argument("--disable-gpu")
#options.add_argument("--no-sandbox")
#options.add_argument("--disable-software-rasterizer")
options.add_argument("--mute-audio")
options.add_argument("--window-size=800,600")
options.add_argument("--disable-popup-blocking")
#options.add_argument("--disable-extensions")

# Open Chrome browser and access Dino game
browser = webdriver.Chrome(options=options)

test_image_dir = "C:/Git/Chrome_AI/src/results/before" # Replace with your location
test_image_name = "Afbeelding2.png"
result_image_dir = "C:/Git/Chrome_AI/src/results/after" # Replace with your location

pretrained = True 
save_export_file = False #inverted logic for some reason
epochs = 20

pretrained_path = "C:/Git/Chrome_AI/src/results/model/default_model/custom_model.pt" #which model will be used if pretrained is True
mode_save_name = "trained.pt" #where the model will be saved 

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

def model_load():
    if not pretrained:
        model = YOLO("yolo11n.pt")
    else:
        model = YOLO(model=pretrained_path)
    return model

# If training is enabled 
def train_model(model):
    if not pretrained:
        #training_results = model.train(data="Fles_dataset.yaml", epochs=epochs, imgsz=640)
        training_results = model.train(data="datainfo.yaml", epochs=epochs, imgsz=640)
        validation_results = model.val()  # Separate validation results
        return training_results, validation_results
    
# If export is enabled
def save_export(model):
    if save_export_file:
        model.save(mode_save_name)
        model.export(format="onnx", opset=11)

def model_test_frame(model, frame):
    # Convert the frame to the format expected by the YOLO model
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # YOLO expects RGB images
    results = model(frame)  # Pass the frame directly to the model
    return results

# Will test the model on image provided 
def model_test(model):
    result = model(test_image_dir + "/" + test_image_name)
    result[0].save(result_image_dir + "/" + test_image_name)

try:
    print("Debug: Opening Browser...")
    browser.get('https://chromedino.com')

    print("debug: Navigated to the Dino game.")
    time.sleep(2)

    print("Debug: Attempting to find canvas...")
    # Find the game canvas element
    canvas = browser.find_element(By.TAG_NAME, 'body')
    print("Debug: Canvas found, send key to start game...")

    # Start the game by sending the SPACE key
    canvas.send_keys(Keys.SPACE)

    print("Debug: Starting game")

    w, h = pyautogui.size()
    print("PIL Screen Capture Speed Test")
    print("Screen Resolution: " + str(w) + 'x' + str(h))

    img = None
    t0 = time.time()
    n_frames = 1
    monitor = {"top": 0, "left": 0, "width": w, "height": h}

    model = model_load()

    with mss.mss() as sct:
        while True:
            img = sct.grab(monitor)
            img = np.array(img)

            results = model_test_frame(model, img)

            # Extract boxes from YOLO results
            detected_boxes = results[0].boxes.xyxy.cpu().numpy()  # Get [x1, y1, x2, y2] format

            # Calculate distances between each pair of boxes
            num_boxes = len(detected_boxes)
            for i in range(num_boxes):
                for j in range(i + 1, num_boxes):
                    box1 = detected_boxes[i]
                    box2 = detected_boxes[j]
                    distance = calculate_distance(box1, box2)
                    print(f"Distance between Box {i} and Box {j}: {distance}")

            # Visualization and key events
            result_image = results[0].plot()
            cv.imshow("Computer Vision", cv.resize(result_image, (0, 0), fx=0.5, fy=0.5))

            key = cv.waitKey(1)
            if key == ord('q'):
                break
            
            elapsed_time = time.time() - t0
            avg_fps = (n_frames / elapsed_time)
            print("Average FPS: " + str(avg_fps))
            n_frames += 1

except Exception as e:
    # Catch and print any errors that occur
    print(f"Debug: An error occurred: {e}")
finally:
    print("Debug: Closing the browser.")
    browser.quit()