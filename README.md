# Real-Time Facial Emotion Recognition Framework

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-blue?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

## 🔬 Research Objective
This repository implements a high-performance computer vision pipeline for **Real-Time Facial Expression Recognition (FER)**. The primary objective was to engineer a latency-optimized system capable of detecting and classifying human emotions in live video streams for Human-Computer Interaction (HCI) and behavioral analysis applications.

Unlike standard implementations, this framework leverages a hybrid **CNN-CRNN architecture** to capture both spatial features and temporal dependencies, ensuring robust classification across 7 distinct emotional states: *Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral*.

## 🛠️ System Architecture

### 1. The Inference Pipeline
The system processes video feeds in a three-stage pipeline designed for sub-50ms latency:
1.  **Face Localization:** Uses optimized Haar Cascade Classifiers (or MTCNN) to detect face bounding boxes in real-time.
2.  **Preprocessing:** Detected faces are dynamically cropped, converted to grayscale, and normalized to `(48, 48, 1)` tensors.
3.  **Classification:** The normalized tensor is passed through a custom **Convolutional Neural Network (CNN)** (and experimental CRNN variants) to output the softmax probability distribution for each emotion.

### 2. Tech Stack
* **Deep Learning:** TensorFlow, Keras (Sequential API)
* **Image Processing:** OpenCV (`cv2`) for frame manipulation.
* **Data Handling:** NumPy, Pandas.
* **Containerization:** Docker for reproducible inference environments.

## 📂 Repository Structure

| Directory/File | Description |
| :--- | :--- |
| `models/` | Stores pre-trained model weights (`.h5` and `.json` architectures). |
| `emotion_analyzer/` | Core inference logic containing the emotion classification class. |
| `training/` | Scripts for data augmentation and model training pipelines. |
| `data/` | Dataset directory (compatible with FER-2013 structure). |
| `video_main.py` | **Main Entry Point:** Launches the webcam feed and runs the real-time analyzer. |
| `Dockerfile` | Configuration for building the production image. |

## 🚀 Quick Start

### Prerequisites
* Python 3.8 or higher
* A working webcam (for live inference)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/HarshaVardhan3002/Face_AI.git](https://github.com/HarshaVardhan3002/Face_AI.git)
    cd Face_AI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Real-Time Inference:**
    ```bash
    python video_main.py
    ```

### 🐳 Running via Docker (Recommended)
To run the application in an isolated environment:
```bash
# Build the image
docker build -t face-ai-research .

# Run container with camera access
docker run --device=/dev/video0:/dev/video0 face-ai-research
