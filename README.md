# YOLOv8-ObjectTracker

A real-time object detection and tracking application using YOLOv8, OpenCV, and CVZone. This project detects objects from a video feed or webcam and draws bounding boxes with confidence scores around the detected objects.

## Features

- **Real-Time Object Detection**: Uses YOLOv8 for accurate object detection in real-time.
- **Object Tracking**: Visualizes object bounding boxes and class labels.
- **Customizable Classes**: Predefined COCO dataset classes.
- **Video/Live Feed**: Supports webcam or video file inputs.
  
## Demo

![YOLOv8 Object Tracker](demo.gif)

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- CVZone

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/alihassanml/YOLOv8-ObjectTracker.git
   cd YOLOv8-ObjectTracker
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download YOLOv8 weights:

   Place the YOLOv8 model weights (`yolov8n.pt`) in the `weights/` directory.

## Usage

Run the following command to start object detection with a webcam:

```bash
python app.py
```

### Adjusting the Detection Classes

You can customize the object detection classes by modifying the `classNames` list in the code.

## Code Explanation

The main components of this project include:

- **YOLOv8**: Used for object detection.
- **OpenCV**: To handle video input and display.
- **CVZone**: For better visualization of bounding boxes and labels.

```python
import os
import cv2
import math
import cvzone
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('weights/yolov8n.pt')

# Predefined COCO classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", ...]
```

## Contributing

Feel free to submit issues, fork the repository, and make pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
