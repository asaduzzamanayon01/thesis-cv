import numpy as np
import os
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("E:\object_detection\Videos\Thesis-1.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8mE.pt")

# classNames_2 = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush", "rickshaw"
#               ]

classNames = [
    'person',
    'rickshaw',
    'rickshaw van',
    'auto rickshaw',
    'truck',
    'pickup truck',
    'private car',
    'motorcycle',
    'bicycle',
    'bus',
    'micro bus',
    'covered van',
    'human hauler'
]

mask = cv2.imread("Mask6.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits_1 = [945, 0, 945, 1440]
limits_2 = [1160, 0, 1160, 1440]
limits_3 = [980, 0, 980, 1440]
jaywalk_totalCount = []
not_jaywalk_totalCount = []
initial_jaywalk_count = 0

# Create directories for saving screenshots if they don't exist
if not os.path.exists('jaywalk_screenshots'):
    os.makedirs('jaywalk_screenshots')

if not os.path.exists('not_jaywalk_screenshots'):
    os.makedirs('not_jaywalk_screenshots')

# Initialize counts
prev_jaywalk_count = 0
prev_not_jaywalk_count = 0

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits_1[0], limits_1[1]), (limits_1[2], limits_1[3]), (0, 0, 255), 5)
    cv2.line(img, (limits_2[0], limits_2[1]), (limits_2[2], limits_2[3]), (0, 255, 0), 5)
    cv2.line(img, (limits_3[0], limits_3[1]), (limits_3[2], limits_3[3]), (0, 255, 0), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 12, (255, 0, 255), cv2.FILLED)

        if limits_3[1] < cy < limits_3[3] and limits_3[0] - 15 < cx < limits_3[2] + 15:
            if not_jaywalk_totalCount.count(id) == 0:
                not_jaywalk_totalCount.append(id)
                cv2.line(img, (limits_2[0], limits_2[1]), (limits_2[2], limits_2[3]), (0, 255, 0), 5)

            # Check if jaywalk_totalCount increased, then pop the last element from not_jaywalk_totalCount
            if len(jaywalk_totalCount) > initial_jaywalk_count:
                if not_jaywalk_totalCount:
                    not_jaywalk_totalCount.pop(-1)  # Popping the last element only
                else:
                    print("not_jaywalk_totalCount is empty. Cannot pop.")
                initial_jaywalk_count += 1

        elif limits_2[1] < cy < limits_2[3] and limits_2[0] - 15 < cx < limits_2[2] + 15:
            if not_jaywalk_totalCount.count(id) == 0:
                not_jaywalk_totalCount.append(id)
                cv2.line(img, (limits_2[0], limits_2[1]), (limits_2[2], limits_2[3]), (0, 255, 0), 5)

            # Check if jaywalk_totalCount increased, then pop the last element from not_jaywalk_totalCount
            if len(jaywalk_totalCount) > initial_jaywalk_count:
                if not_jaywalk_totalCount:
                    not_jaywalk_totalCount.pop(-1)  # Popping the last element only
                else:
                    print("not_jaywalk_totalCount is empty. Cannot pop.")
                initial_jaywalk_count += 1

        elif limits_1[1] < cy < limits_1[3] and limits_1[0] - 15 < cx < limits_1[2] + 15:
            if jaywalk_totalCount.count(id) == 0:
                jaywalk_totalCount.append(id)
                cv2.line(img, (limits_1[0], limits_1[1]), (limits_1[2], limits_1[3]), (255, 0, 0), 5)

    cvzone.putTextRect(img, f' Jaywalk Count: {len(jaywalk_totalCount)}', (50, 50))
    cvzone.putTextRect(img, f' Not Jaywalk Count: {len(not_jaywalk_totalCount)}', (50, 120))
    # cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Check if jaywalk count has increased
    if len(jaywalk_totalCount) > prev_jaywalk_count:
        cv2.imwrite(f'jaywalk_screenshots/jaywalk_{len(jaywalk_totalCount)}.png', img)
        prev_jaywalk_count = len(jaywalk_totalCount)

    # Check if not jaywalk count has increased
    if len(not_jaywalk_totalCount) > prev_not_jaywalk_count:
        cv2.imwrite(f'not_jaywalk_screenshots/not_jaywalk_{len(not_jaywalk_totalCount)}.png', img)
        prev_not_jaywalk_count = len(not_jaywalk_totalCount)

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
