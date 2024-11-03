# Car Race Object Detection and Tracking

This repository contains code for detecting and tracking cars in a video using YOLO (You Only Look Once) object detection and the SORT (Simple Online and Realtime Tracking) algorithm.

## Overview

The project performs the following tasks:

1. **Object Detection**:
   - Utilizes the YOLO model to detect cars in each frame of a given video.
   - Filters out car detections based on confidence scores.

2. **Object Tracking**:
   - Applies the SORT algorithm to track detected cars across frames.
   - Assigns unique IDs to each tracked car.

3. **Car Counting**:
   - Sets up virtual counting lines in the video.
   - Increments the total car count when a car crosses the counting line.

## Requirements

- Python 3.x
- OpenCV (for video processing)
- Ultralytics YOLO (for object detection)
- SORT (Simple Online and Realtime Tracking)

## Usage

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/car-race-object-detection.git
   cd car-race-object-detection
