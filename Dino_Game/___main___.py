from os import access
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from screeninfo import get_monitors
from ultralytics import YOLO

monitor_used = 1
sct_mon = monitor_used + 1

import numpy as np
import cv2
from mss import mss
from PIL import Image

options = Options()
options.binary_location = "/home/oliwier-desktop/Downloads/firefox/firefox"
options.set_preference("browser.download.folderList",2)
options.set_preference("browser.fullscreen.autohide", True)
options.set_preference("browser.download.manager.showWhenStarting", False)
options.set_preference("browser.download.dir","/Data")
options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream,application/vnd.ms-excel")
driver = webdriver.Firefox(options=options)
driver.maximize_window();
driver.get("https://chromedino.com/")
assert "T-Rex Dinosaur Game" in driver.title

#elem = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME,"fc-button-label")))   
elem = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CLASS_NAME,"fc-button")))    
elem = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME,"fc-button")))    
print("element is clickable")

action = ActionChains(driver)

consent_btn = driver.find_element(By.CLASS_NAME, "fc-button")

action.move_to_element(consent_btn)
action.click(consent_btn)
action.perform()

runner = driver.find_element(By.CLASS_NAME, "runner-container")

#action.move_to_element(runner)
action.send_keys(" ")
action.perform()


monitors = get_monitors()



#print(f"Monitor {monitors[monitor_used].name}: {monitors[monitor_used].width}x{monitors[monitor_used].height}")
print(  type(monitors[monitor_used].width) )
with mss() as sct:
    mon1 = sct.monitors[sct_mon]
    dis = {'left': mon1["left"]+(int)(monitors[monitor_used].width/3.275), 'top': mon1["top"]+(int)(monitors[monitor_used].height/7.2), 'width': (int)(monitors[monitor_used].width/2.56), 'height': (int)(monitors[monitor_used].height/8.2)}
    
    model_path = '/home/oliwier-desktop/Projects/Active/Chrome_AI/Model/custom_model_acht.pt'
    model = YOLO(model_path)
    while True:
        
        screenShot = sct.grab(dis) 
        img = Image.frombytes(
            'RGB', 
            (screenShot.width, screenShot.height), 
            screenShot.rgb, 
        )
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        resized_image = cv2.resize(img, (640, 640))
        results = model(img)
        annotated_image = results[0].plot()
       
        # Display the result
        cv2.imshow("YOLO Inference", annotated_image)
        
        #cv2.imshow('test', np.array(img))
        cv2.moveWindow("dino game inference", (int)(monitors[monitor_used].width/5),(int)(monitors[monitor_used].height/9.2))
        if cv2.waitKey(33) & 0xFF in (
            ord('q'), 
            27, 
        ):
            break
        
driver.close()
