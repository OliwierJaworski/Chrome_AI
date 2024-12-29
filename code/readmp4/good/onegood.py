import cv2
import numpy as np
from ultralytics import YOLO
import threading
import math 

pretrained_path = "C:/DERDEJAAR/SmartSystem/githubown/Chrome_AI/src/results/model/default_model/custom_model_acht.pt"  

class DetectionThread(threading.Thread):
    def __init__(self, model, frame):
        threading.Thread.__init__(self)
        self.model = model
        self.frame = frame
        self.results = None

    def run(self):
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  
        self.results = self.model(frame_rgb) 

# Load the YOLO model
def model_load():
    model = YOLO(pretrained_path)
    return model

try:
    model = model_load()
    model.overrides['verbose'] = False  # Disable internal logging
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Open the video file
video_path = "C:/DERDEJAAR/SmartSystem/githubown/Chrome_AI/gameplay/gamePlay.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

in_jump_mode = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video file")
        break

    detection_thread = DetectionThread(model, frame)
    detection_thread.start()
    detection_thread.join() 

    dino_box = None
    cactus_boxes = []

    if detection_thread.results:
        for result in detection_thread.results:
            boxes = result.boxes  # Detected boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                label = result.names[int(box.cls[0])]  
                confidence = box.conf[0]  

                if label == "Dino":
                    dino_box = (x1, y1, x2, y2)
                elif label == "Cactus":
                    cactus_boxes.append((x1, y1, x2, y2))

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    # Calculate and display the distance to the nearest cactus
    if dino_box and cactus_boxes:
        dino_center = ((dino_box[0] + dino_box[2]) // 2, (dino_box[1] + dino_box[3]) // 2)
        nearest_distance = float("inf")
        nearest_cactus = None

        for cactus_box in cactus_boxes:
            cactus_center = ((cactus_box[0] + cactus_box[2]) // 2, (cactus_box[1] + cactus_box[3]) // 2)
            # Calculate the Euclidean distance
            distance = math.sqrt((dino_center[0] - cactus_center[0]) ** 2 + (dino_center[1] - cactus_center[1]) ** 2)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_cactus = cactus_center

        if nearest_distance <= 210 and not in_jump_mode:
            print("Jump") 
            in_jump_mode = True  

        if nearest_distance > 210:
            in_jump_mode = False 

        if nearest_cactus:
            cv2.line(frame, dino_center, nearest_cactus, (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Distance: {int(nearest_distance)} px",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
