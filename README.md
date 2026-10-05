<div align="center">

# 🧠 Project 4 — Image & Text Recognition

### A Python AI pipeline for OCR and real-time object detection

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://github.com/tesseract-ocr/tesseract)
[![MobileNet](https://img.shields.io/badge/MobileNet-SSD-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://github.com/chuanqi305/MobileNet-SSD)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

A production-ready AI recognition pipeline that processes raw visual data through **two independent paths**:

| Path | Technology | Output |
|------|-----------|--------|
| 🔤 **OCR** | Tesseract + PyTesseract | Extracted text from images |
| 🎯 **Object Detection** | MobileNet-SSD via OpenCV DNN | Labeled bounding boxes |

---

## 🎯 Objectives

- ✅ Image representation as matrix `(height × width × channels)`
- ✅ Preprocessing pipeline: grayscale → blur → deskew → threshold
- ✅ OCR with configurable Page Segmentation Mode (PSM)
- ✅ Object detection with bounding boxes
- ✅ Confidence filtering `(≥ 80%)`
- ✅ Visual output saved to `output/`

---

## 🏗️ Architecture

```
                        ┌─────────────────┐
                        │   INPUT IMAGE   │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  IMAGE LOADING  │
                        │    (OpenCV)     │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  PREPROCESSING  │
                        └────────┬────────┘
                                 │
               ┌─────────────────┴─────────────────┐
               │                                   │
               ▼                                   ▼
      ┌─────────────────┐               ┌─────────────────────┐
      │    OCR PATH     │               │  OBJECT DETECTION   │
      │─────────────────│               │─────────────────────│
      │  Grayscale      │               │  blobFromImage()    │
      │  Gaussian Blur  │               │  MobileNet-SSD      │
      │  Deskew         │               │  Raw Detections     │
      │  Threshold      │               │  Decode + Filter    │
      │  PyTesseract    │               │  Bounding Boxes     │
      └────────┬────────┘               └──────────┬──────────┘
               │                                   │
               ▼                                   ▼
         Recognized Text                   Objects (≥ 80%)
               │                                   │
               └─────────────────┬─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  FINAL OUTPUT   │
                        │  output/ dir    │
                        └─────────────────┘
```

---

## 🛠️ Technologies

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.8+ | Primary language |
| **OpenCV** (`cv2`) | 4.x | Image loading, preprocessing, DNN inference, visualization |
| **PyTesseract** | latest | Python wrapper for Tesseract OCR |
| **Tesseract OCR** | 4.x+ | OCR engine executable |
| **MobileNet-SSD** | Caffe | Pre-trained object detection model |
| **NumPy** | latest | Array operations |
| **Pillow** | latest | Image format support for PyTesseract |

---

## 📁 Project Structure

```
project-root/
│
├── 📄 app.py                   ← Main application entry point
├── 📄 requirements.txt         ← Python dependencies
├── 📄 README.md
├── 📄 .gitignore
├── 📄 download_models.py       ← Downloads model files
├── 📄 build_demo_model.py      ← Builds demo model for testing
│
├── 📂 input/
│   └── sample.jpg              ← Place your test images here
│
├── 📂 output/
│   ├── grayscale.jpg           ← Step 1 intermediate
│   ├── blurred.jpg             ← Step 2 intermediate
│   ├── deskewed.jpg            ← Step 3 intermediate
│   ├── thresholded.jpg         ← Step 4 intermediate
│   ├── ocr_result.txt          ← OCR recognized text
│   └── object_detection_result.jpg  ← Annotated detection image
│
├── 📂 models/
│   ├── MobileNetSSD_deploy.prototxt    ← Architecture (auto-downloaded)
│   └── MobileNetSSD_deploy.caffemodel  ← Weights (~22 MB)
│
├── 📂 src/
│   ├── __init__.py
│   ├── image_loader.py         ← OpenCV image loading + validation
│   ├── preprocessing.py        ← Grayscale, blur, deskew, threshold
│   ├── ocr.py                  ← PyTesseract OCR integration
│   ├── object_detection.py     ← MobileNet-SSD inference + confidence filter
│   ├── visualization.py        ← Bounding box drawing
│   └── utils.py                ← Shared constants and helpers
│
└── 📂 tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_ocr.py
    └── test_object_detection.py
```

---

## ⚙️ Installation

### 1️⃣ Clone / open the project

```bash
cd "Image or Text Recognition"
```

### 2️⃣ Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install Tesseract OCR

> Required for the OCR path. PyTesseract is only a wrapper — the engine must be installed separately.

<details>
<summary>🪟 Windows</summary>

Download and install from:
👉 https://github.com/UB-Mannheim/tesseract/wiki

Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe` *(auto-detected)*

If installed elsewhere:
```bash
set TESSERACT_CMD=C:\path\to\tesseract.exe
```
</details>

<details>
<summary>🍎 macOS</summary>

```bash
brew install tesseract
```
</details>

<details>
<summary>🐧 Ubuntu / Debian</summary>

```bash
sudo apt install tesseract-ocr
```
</details>

### 4️⃣ MobileNet-SSD Model Setup

**Option A — Real model (~22 MB) for actual detection:**
```bash
python download_models.py
```
Then download `MobileNetSSD_deploy.caffemodel` from:
👉 https://github.com/chuanqi305/MobileNet-SSD → place in `models/`

**Option B — Demo model for pipeline testing:**
```bash
python build_demo_model.py
```
> ⚠️ Demo model uses random weights. Replace with the real model for accurate detections.

---

## 🚀 How to Run

```bash
# Both OCR + Object Detection (recommended)
python app.py --image input/sample.jpg --mode both

# OCR only
python app.py --image input/sample.jpg --mode ocr

# Object detection only
python app.py --image input/sample.jpg --mode detection

# Custom confidence threshold
python app.py --image input/sample.jpg --mode detection --confidence 0.80

# Custom Tesseract PSM
python app.py --image input/sample.jpg --mode ocr --psm 6
```

---

## 🔤 OCR — Page Segmentation Modes

| PSM | Description | Best For |
|-----|-------------|----------|
| `3` | Fully automatic | General documents |
| `6` | Single uniform block | Paragraphs, forms *(default)* |
| `7` | Single text line | One-line images |
| `11` | Sparse text | Scattered text |

---

## 🎯 Object Detection — Supported Classes

The MobileNet-SSD model detects **20 PASCAL VOC classes**:

```
aeroplane  bicycle    bird       boat       bottle
bus        car        cat        chair      cow
diningtable  dog      horse      motorbike  person
pottedplant  sheep    sofa       train      tvmonitor
```

---

## 📊 Confidence Threshold

> Default: **80%** — configurable at runtime

```
Detection Score    Status
─────────────────────────────────
  ≥ 80%          ✅  ACCEPTED  →  drawn on output image
  < 80%          ❌  REJECTED  →  not shown
```

```bash
# Override threshold
python app.py --image input/sample.jpg --mode detection --confidence 0.75
```

---

## 🔬 Preprocessing Pipeline

| Step | Function | Purpose |
|------|----------|---------|
| 1️⃣ Grayscale | `convert_to_grayscale()` | Reduce to single channel |
| 2️⃣ Gaussian Blur | `apply_gaussian_blur()` | Remove noise |
| 3️⃣ Deskew | `deskew()` | Align text horizontally |
| 4️⃣ Adaptive Threshold | `apply_adaptive_threshold()` | Binary image for OCR |

> All intermediate images are saved to `output/` for visual inspection.

---

## 📋 Sample Output

```
==================================================
 PROJECT 4 — IMAGE & TEXT RECOGNITION
==================================================

  Mode       : BOTH
  Image      : input/sample.jpg
  Confidence : 80%
  PSM        : 6

Image Information:
  Width    : 800 px
  Height   : 400 px
  Channels : 3  (BGR color)

--------------------------------------------------
 OCR PIPELINE
--------------------------------------------------
  [1/4] Grayscale ................. SUCCESS
  [2/4] Gaussian Blur ............. SUCCESS
  [3/4] Deskew .................... SUCCESS
  [4/4] Thresholding .............. SUCCESS

================== OCR RESULT ==================

Recognized Text:
Student Name: Nanda Kishore
Roll Number: 12345
Project: Artificial Intelligence

  OCR result saved: output/ocr_result.txt

--------------------------------------------------
 OBJECT DETECTION PIPELINE
--------------------------------------------------
  Model loaded .................. SUCCESS
  Blob creation ................. SUCCESS
  Inference ..................... SUCCESS

  Detected objects:

  Person           91.40%  ACCEPTED
  Car              87.80%  ACCEPTED
  Dog              61.20%  REJECTED

  Detection image saved: output/object_detection_result.jpg

==================================================
 VALIDATION
==================================================

  Library Integration .......... PASS
  Preprocessing Integrity ...... PASS
  Confidence Benchmark ......... PASS
  Visual Confirmation .......... PASS

==================================================
 PROJECT 4 STATUS: COMPLETED
==================================================
```

---

## ✅ Validation Checklist

### Milestone 1 — Library Integration
- [x] OpenCV loads and processes images
- [x] PyTesseract wrapper installed
- [x] Tesseract executable detected
- [x] MobileNet-SSD loads via `cv2.dnn.readNetFromCaffe()`

### Milestone 2 — Preprocessing Integrity
- [x] Grayscale conversion → `output/grayscale.jpg`
- [x] Gaussian blur → `output/blurred.jpg`
- [x] Deskewing → `output/deskewed.jpg`
- [x] Adaptive thresholding → `output/thresholded.jpg`

### Milestone 3 — Accuracy Benchmarking
- [x] Confidence threshold = 80%
- [x] Detections below 80% are rejected
- [x] Accepted detections are ≥ 80%
- [x] OCR produces readable text on suitable images

### Milestone 4 — Visual Confirmation
- [x] OCR output saved to `output/ocr_result.txt`
- [x] Detection output saved to `output/object_detection_result.jpg`
- [x] Bounding boxes drawn on detected objects
- [x] Labels and confidence percentages visible

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

> Expected: **24 passed, 2 skipped** *(OCR tests skip if Tesseract is not installed)*

---

## ⚠️ Limitations

| # | Limitation |
|---|-----------|
| 1 | Tesseract must be installed separately — PyTesseract is only a wrapper |
| 2 | MobileNet-SSD caffemodel (~22 MB) must be downloaded manually |
| 3 | Demo model produces random detections — replace with real caffemodel |
| 4 | OCR accuracy depends on image quality — best on clean, high-contrast images |
| 5 | MobileNet-SSD detects only 20 PASCAL VOC classes |

---

## 🔮 Future Improvements

- [ ] 🚀 Add YOLOv8 as an alternative detection backend
- [ ] 📄 Add PDF input support for OCR
- [ ] 🗂️ Add batch processing for multiple images
- [ ] 🌐 Add a simple web UI (Flask / Streamlit)
- [ ] ⚡ Add GPU acceleration via OpenCV CUDA backend
- [ ] 🌍 Add multi-language support for Tesseract
- [ ] 🔲 Add NMS (Non-Maximum Suppression) for overlapping detections
- [ ] 📊 Add confidence score histogram visualization

---

<div align="center">

Made with ❤️ using Python, OpenCV, and Tesseract

</div>
