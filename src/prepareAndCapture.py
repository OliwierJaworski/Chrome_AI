# pip install selenium
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Capture
import mss
from PIL import Image, ImageGrab
import pyautogui

import cv2 as cv
import numpy as np
import time

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
    with mss.mss() as sct:
        while True:
            img = sct.grab(monitor)
            img = np.array(img)                         # Convert to NumPy array
            # img = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # Convert RGB to BGR color
            
            small = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
            cv.imshow("Computer Vision", small)

            # Break loop and end test
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