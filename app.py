import os
os.chdir('yolov10')


import cv2
import math
import cvzone
from ultralytics import YOLO


model = YOLO('weights/yolov8n.pt') 


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


cap = cv2.VideoCapture(0)
while True:
    res,image = cap.read()
    model = YOLO('weights/yolov8n.pt')  
    result = model(image)[0]
    
    # Normalize The Result
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        
        cvzone.cornerRect(image, (x1, y1, w, h))
        
        center_x = x1 + w // 2
        center_y = y1 - 15  
        confidence = math.ceil(box.conf[0]*100)/100
        
        name = int(box.cls[0])
        text = classNames[name]
        text = f'{text} {confidence}'
        font_scale = 3
        thickness = 2
        font = cv2.FONT_HERSHEY_PLAIN
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        
        adjusted_x = center_x - text_size[0] // 2
        
        
        img, bbox = cvzone.putTextRect(
            image, text, (adjusted_x, center_y),  
            scale=font_scale, thickness=thickness, 
            colorT=(255, 255, 255), colorR=(0, 255, 0),
            font=font, 
            offset=10,  
            border=2, colorB=(0, 117, 0)  
        )
    cv2.namedWindow('example',cv2.WINDOW_NORMAL)
    cv2.imshow('example', image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    