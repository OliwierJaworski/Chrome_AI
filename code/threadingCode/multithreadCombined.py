import threading
import queue
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
from ultralytics import YOLO
import cv2 as cv
import numpy as np
import math
import mss
import pyautogui

# Path to your ChromeDriver
CHROME_DRIVER_PATH = 'C:\\Program Files\\chrome-win64\\chrome-win64\\chrome.exe'

# Start WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--mute-audio")
options.add_argument("--window-size=800,600")
options.add_argument("--disable-popup-blocking")
browser = webdriver.Chrome(options=options)

# Define YOLO model
pretrained_path = "C:/Git/Chrome_AI/src/results/model/default_model/custom_model_acht.pt"

def model_load():
    model = YOLO(model=pretrained_path)
    return model

# Helper function to calculate distance
def calculate_distance(box1, box2):
    centroid1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    centroid2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
    return distance

# Initialize global variables
frame_queue = queue.Queue(maxsize=5)  # Smaller queue to avoid buildup
processed_frame_queue = queue.Queue(maxsize=5)  # Separate queue for YOLO results
stop_threads = threading.Event()

def capture_screen(monitor, frame_queue):
    """Thread for capturing screen frames."""
    with mss.mss() as sct:
        while not stop_threads.is_set():
            img = sct.grab(monitor)
            frame = np.array(img)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Convert RGB to BGR
            
            # Downscale frame to speed up processing
            downscaled_frame = cv.resize(frame, (640, 360))  # Further reduced resolution
            try:
                frame_queue.put_nowait(downscaled_frame)
            except queue.Full:
                print("Capture thread: Frame queue is full, dropping frame.")

def process_frames(model, frame_queue, processed_frame_queue):
    """Thread for processing frames with the YOLO model."""
    while not stop_threads.is_set():
        try:
            frame = frame_queue.get_nowait()  # Non-blocking
            results = model(frame)  # YOLO processing

            try:
                processed_frame_queue.put_nowait(results[0])
            except queue.Full:
                print("Processing thread: Processed frame queue is full, dropping result.")

        except queue.Empty:
            continue

def main():
    try:
        print("Debug: Opening Browser...")
        browser.get('https://chromedino.com')

        print("Debug: Navigated to the Dino game.")
        time.sleep(2)

        # Find the game canvas element
        canvas = browser.find_element(By.TAG_NAME, 'body')
        print("Debug: Canvas found, sending key to start game...")

        # Start the game by sending the SPACE key
        canvas.send_keys(Keys.SPACE)

        print("Debug: Starting game")

        # Define capture region for the game area only
        monitor = {"top": 200, "left": 300, "width": 800, "height": 400}  # Adjust these dimensions

        # Load the YOLO model
        model = model_load()

        # Create and start threads
        capture_thread = threading.Thread(target=capture_screen, args=(monitor, frame_queue), daemon=True)
        process_thread = threading.Thread(target=process_frames, args=(model, frame_queue, processed_frame_queue), daemon=True)

        capture_thread.start()
        process_thread.start()

        # Display frames in the main thread
        while not stop_threads.is_set():
            if not processed_frame_queue.empty():
                results = processed_frame_queue.get_nowait()
                result_image = results.plot()

                # Show the frame
                cv.imshow("Computer Vision", cv.resize(result_image, (800, 450)))

                # Check for user quit signal
                if cv.waitKey(1) == ord('q'):
                    stop_threads.set()
                    break

    except Exception as e:
        print(f"Debug: An error occurred: {e}")
    finally:
        print("Debug: Closing the browser.")
        stop_threads.set()
        cv.destroyAllWindows()
        browser.quit()

if __name__ == "__main__":
    main()
