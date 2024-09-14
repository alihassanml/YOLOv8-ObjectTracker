import os
os.chdir('yolov10') 


import cv2
import math
import cvzone
from ultralytics import YOLO


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


image_path = '../KOA_Nassau_2697x1517.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error loading image!")
else:
    # Load Model
    model = YOLO('weights/yolov8n.pt')  

    # Run the model on the image
    result = model(image)[0]
    
    # Process the results
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
            colorT=(255, 255, 255), colorR=(255, 0, 255),
            font=font, 
            offset=10,  
            border=3, colorB=(0, 255, 0)  
        )

    # Display the image
    cv2.namedWindow('example', cv2.WINDOW_NORMAL)
    cv2.imshow("example", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
