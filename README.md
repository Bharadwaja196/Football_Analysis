<div align="center">
  <h1>⚽ Advanced Football Tracking & Analytics Engine</h1>
  <p><strong>A professional Computer Vision pipeline engineered to extract real-world physics, track players under heavy occlusion, dynamically cluster team teams, and mathematically evaluate YOLO detection models natively.</strong></p>

  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/OpenCV-4.5+-green.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv5x-orange.svg" alt="YOLOv5">
  <img src="https://img.shields.io/badge/Machine_Learning-PyTorch-red.svg" alt="PyTorch">
</div>

---

## 🌟 Project Achievements & Core Systems

This project transcends basic object detection by introducing a multi-layered analytical pipeline explicitly built for sports intelligence. 

### 1. 🎯 Precision Object Detection (YOLO)
Utilizing a custom-trained **YOLOv5x** model, the system is designed to detect extremely small and fast-moving objects (like a football in flight) while maintaining strong confidence intervals on players, the goalkeeper, and referees even through intense motion blur.

### 2. 🤖 Robust Temporal Tracking (ByteTrack / SORT)
Sports tracking is notoriously difficult because players constantly tackle, cross paths, and occlude each other. This repository natively integrates ByteTrack to drastically minimize **ID Switch Rates**, ensuring a player maintains their exact tracking ID from the start of the play to the end.

### 3. 🎨 K-Means Team Assignment
Instead of manually hardcoding team colors, the system utilizes **K-Means Clustering** to dynamically scrape pixel data from isolated player bounding boxes. It then autonomously splits the players into two distinct cohesive tactical units based on jersey color.

### 4. 📐 Homography & Perspective Transformation
It is physically impossible to measure distance from a 2D broadcast camera angle. This project uses **View Transformers** to map the raw angled video output onto a flat, top-down 2D tactical plane (Homography). 

### 5. ⚡ Physics Engine (Speed & Distance)
Because we use perspective transformation to convert pixels into real-world meters, we can calculate physics. The pipeline tracks bounding box centroids across sequential frames to extract real-world **Distance Covered (m)** and exact **Running Speed (km/h)** for every single player.

### 6. 📊 Dynamic ML Evaluation Module
A professional Machine Learning architecture must be verifiable. We engineered a native `evaluate.py` wrapper that automatically loads ground-truth test data and calculates rigid metrics:
*   **mAP (Mean Average Precision):** Evaluates bounding-box quality at different confidence thresholds.
*   **Detection Accuracy / F1-Score Proxy:** Automatically spits out the true mathematical Precision and Recall of the YOLO detections.

### 7. 🛡️ Hardened Git Architecture
This project features a specially calibrated `.gitignore` to protect CI/CD pipelines and GitHub restrictions by strictly blocking massive `models/best.pt` arrays (>100MB), raw `.mp4` broadcast videos, and local heavy cache files.

---

## 🚀 How to Install and Run Locally

Because the heavy ML models and videos are correctly ignored from GitHub, you will need to add those locally after cloning the repository. Follow these exact steps to run the pipeline on your own machine!

### Step 1: Clone the Repository
Open your terminal and clone this repository to your computer:
```bash
git clone https://github.com/Bharadwaja196/Football_Analysis.git
cd Football_Analysis
```

### Step 2: Install the Requirements
It is highly recommended to use a Conda environment or standard Python virtual environment. Install all the necessary computer vision dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Add the YOLO Model
Because GitHub blocks files over 100MB, the custom YOLO model is not in this repository. 
1. Open the `models/` folder.
2. Drop your trained custom weights file into this folder and name it `best.pt`. *(Path: `models/best.pt`)*

### Step 4: Add Your Match Video
1. Open the `input_videos/` folder.
2. Drop the raw broadcast broadcast clip you wish to analyze into this folder and name it `08fd33_4.mp4` (or simply edit `main.py` line 13 to point to your new video name).

### Step 5: Execute the Pipeline!
With your model and video in place, the entire multi-layered pipeline (tracking, physics, clustering, and dynamic evaluation) is triggered by a single command:

```bash
python main.py
```

### 📈 Output
1. An annotated video containing all physics metrics and player tracking boxes will be exported directly to `output_videos/output_video.avi`.
2. When the video finishes rendering, your terminal will instantly transition to evaluate the YOLO model on its test dataset, printing your exact **mAP** and **Precision metrics** directly to your screen!
