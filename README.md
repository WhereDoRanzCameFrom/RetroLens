# 📷 Retrolens - Hand Tracking Filter & Portal

Retrolens is an interactive Computer Vision and Digital Image Processing (DIP) project that applies real-time gesture-based camera filters using MediaPipe hand landmark tracking and OpenCV image manipulations.

---

## 📝 Project Overview

The core concept of **Retrolens** is to establish a dynamic "portal" bounded by multi-hand fingertip coordinates in real time. Rather than applying a global filter to the entire camera stream, the application calculates a region of interest (ROI) bounded by detected hand landmarks, performing spatial and color transformations exclusively within or around that bounding box.

This project demonstrates practical applications of:
- **Hand Gesture Recognition & Landmark Detection** via landmark analysis.
- **Dynamic Bounding Box & Polygon Masking** based on real-time spatial vectors.
- **Color Space Transformations & Digital Image Processing** algorithms applied per-frame.

---

## 🛠️ Image Processing & Vision Techniques

Retrolens implements a variety of Digital Image Processing (DIP) filters and algorithms using **OpenCV** and **NumPy**:

- **Spatial Masking & Region of Interest (ROI):** Dynamic calculation of geometric boundaries derived from 2D coordinates of four key fingertips (index fingers and thumbs of both hands).
- **Color Space Manipulation:**
  - **Mono / Grayscale:** Luma transformation for luminance evaluation.
  - **Sepia:** Weighted matrix multiplication across BGR color channels.
  - **Dual-Tone:** Mapping pixel intensity gradients to custom two-color spectrums.
  - **Invert:** Bitwise NOT operations on image matrices.
- **Filter Algorithms & FX:**
  - **Blur:** Spatial filtering using Gaussian and Box convolution kernels.
  - **Pixelate:** Spatial downsampling via interpolation followed by upscale resampling.
  - **Thermal:** Pseudocolor mapping utilizing predefined color lookup tables (LUTs).
  - **Sketch:** Gradient detection using Sobel operators or Canny edge detection combined with adaptive thresholding.
  - **Glitch & Neon:** Channel shifting, edge amplification, and saturation boosting.
  - **Galaxy (Background Segmentation):** Foreground/background separation utilizing deep learning inference via MediaPipe Selfie Segmenter.

---

## 💻 Tech Stack & Dependencies

- **Python 3.7+**
- **OpenCV (`opencv-python`):** Image matrix manipulation, color transformations, dynamic masking, and video stream rendering.
- **MediaPipe (`mediapipe`):** Machine learning pipeline for 21 3D hand landmark detection and real-time selfie segmentation.
- **NumPy (`numpy`):** Fast multi-dimensional array processing and mathematical matrix operations for frame processing.

---

## 📐 Machine Learning Models Used

To ensure robust performance, the application relies on two lightweight pre-trained TFLite/Task models:
1. `hand_landmarker.task` — Multi-hand detection and 3D landmark localization pipeline.
2. `selfie_segmenter.tflite` — Real-time portrait image segmentation model for background extraction.

---

## 📂 Project Structure

```text
├── main.py                     # Main application entry point & video processing loop
├── requirements.txt            # Python dependencies
├── hand_landmarker.task        # MediaPipe Hand Tracking model
└── selfie_segmenter.tflite     # MediaPipe Background Segmentation model
