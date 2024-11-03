from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

model = YOLO("ObjectDetection/yolo-weight/yolov8n.pt")

cap = cv.VideoCapture("ObjectDetection/carRace.mp4")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [0, 400, 1280, 400]

totalCount = []

mask = cv.imread("ObjectDetection/CarCounter/mask2.png")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    imgMask = cv.bitwise_and(frame, mask)
    
    results = model(imgMask, stream=True)
    
    detection = np.empty((0, 5))
    
    graphics = cv.imread("ObjectDetection/CarCounter/graphics.png", cv.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, graphics, (0, 0))
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            conf = math.ceil((box.conf[0]*100))/100
            
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass == "car" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detection = np.vstack((detection, currentArray))
    
    trackerResult = tracker.update(detection)
    # cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (255, 255, 255), 5)
    
    for result in trackerResult:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2-x1, y2-y1
        bbox = (x1, y1, w, h)
        print(result)
        cvzone.cornerRect(frame, bbox)
        cvzone.putTextRect(frame, f"{id}", (max(0, x1), max(35, y1)), scale=2, thickness=1)
        
        cx, cy = x1+w//2, y1+h//2
        cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        
        if limits[0] < cx < limits[2] and limits[1]-15 < cy < limits[1]+15:
            if id not in totalCount:
                totalCount.append(id)
                # cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        
    cv.putText(frame, str(len(totalCount)), (255, 100), cv.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 8)
    
    cv.imshow("Car Race", frame)
  
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

