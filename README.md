# Acoustic Gunshot Detection & Localization System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![DSP](https://img.shields.io/badge/Signal_Processing-Librosa-orange?style=for-the-badge&logo=soundcharts&logoColor=white)](https://librosa.org/)
[![ML](https://img.shields.io/badge/Model-CNN-green?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

## 🔬 Research Objective
This project implements a **Distributed Acoustic Sensor System** designed for the real-time detection and localization of ballistic discharges (gunshots). The system was engineered to address public safety challenges by utilizing an **omnidirectional microphone array** to calculate the **Angle of Arrival (AoA)** of high-impulse acoustic events in noisy urban environments.

Unlike standard audio classifiers, this system integrates **Digital Signal Processing (DSP)** for noise reduction with a **Deep Learning classifier** to distinguish actual gunshots from false positives (fireworks, car backfires, jackhammers) with high precision.

## 🛠️ System Architecture

### 1. Hardware Stack
* **Sensor Node:** Omnidirectional Microphone Array (4-mic configuration).
* **Processing Unit:** Optimized for edge deployment (e.g., Raspberry Pi 4 / Nvidia Jetson).

### 2. Signal Processing Pipeline
The pipeline processes raw audio streams in 3 stages:
1.  **Event Triggering:** Energy-based thresholding detects impulse events to save compute power.
2.  **Feature Extraction:** Converts raw waveforms into **Mel-Frequency Cepstral Coefficients (MFCCs)** and **Log-Mel Spectrograms**.
3.  **Localization:** Uses **Time Difference of Arrival (TDOA)** algorithms across the microphone array to triangulate the source coordinates ($x, y$).

### 3. Machine Learning Model
* **Architecture:** A lightweight Convolutional Neural Network (CNN) trained on spectral features.
* **Dataset:** Trained on a custom dataset of ballistic sounds mixed with urban noise profiles (UrbanSound8K + chemically synthesized gunshot samples).
* **Performance:** Achieves sub-second latency (<200ms) for real-time alert generation.

## 📂 Repository Structure

| Directory | Description |
| :--- | :--- |
| `dsp_algorithms/` | TDOA and GCC-PHAT implementations for source localization. |
| `models/` | Pre-trained `.h5` / `.tflite` models for gunshot classification. |
| `data_processing/` | Scripts for audio normalization and MFCC extraction. |
| `hardware_drivers/` | Interface code for microphone array data acquisition. |
| `realtime_inference.py` | **Main Entry Point:** Runs the continuous listening loop. |

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* `librosa`, `numpy`, `scipy`, `tensorflow` (or `torch`)
* PortAudio (for microphone access)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/HarshaVardhan3002/GunShot_Detection.git](https://github.com/HarshaVardhan3002/GunShot_Detection.git)
    cd GunShot_Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Simulation/Inference:**
    ```bash
    python realtime_inference.py --mic_index 0
    ```

## 📊 Key Results
* **Detection Accuracy:** >92% sensitivity in simulated urban noise environments (SNR 10dB).
* **Localization Error:** <5° mean angular error using TDOA.
* **Latency:** End-to-end processing time of 150ms on edge hardware.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed by [Sai Harshavardhan](https://github.com/HarshaVardhan3002)*
