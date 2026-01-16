# Real-Time Object Boundary Detection

## Description
This project implements a real-time object boundary detection system using classical OpenCV techniques.
Moving objects are detected from a live webcam feed, and boundary around the moving object is obtained using contour merging and Convex Hull to avoid fragmented detections.

## Technologies Used
- Python
- OpenCV
- NumPy

## Techniques Used
- Background Subtraction (MOG2)
- Morphological Operations (Erosion & Dilation)
- Contour Detection & Filtering
- Contour Merging
- Convex Hull

## How to Run
1. Install dependencies
    pip install opencv-python numpy
2. Run the script
    python main.py

# Gaze Controlled Virtual Mouse

## Description
This project implements a gaze-controlled virtual mouse using OpenCV and MediaPipe.
Eye movements are tracked in real time to control cursor motion.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- PyAutoGUI

## How to Run
1. Install dependencies
   pip install -r requirements.txt
2. Run the script
   python gaze_mouse.py
