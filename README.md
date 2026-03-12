<div align="center">

# 🛡️ DeepGuard AI

### Real-Time Deepfake & AI-Generated Media Detection

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**IIT Bombay — Hack & Break: Generative AI & Cybersecurity Innovation Challenge 2026**

[Features](#-features) · [Architecture](#️-system-architecture) · [Installation](#-installation) · [API Docs](#-api-reference)

</div>

---

## 📌 Overview

DeepGuard AI is a professional-grade media authenticity verification platform that detects AI-generated and manipulated content using a **dual-model ensemble** of Vision Transformers (ViT).

Unlike single-model detectors, DeepGuard AI runs **30 inference passes per image** — combining 2 models × 5 augmentations × 3 face crop scales — then merges their votes with adaptive weighting for maximum accuracy.

> **Theme**: Cybersecurity + Generative AI  
> **Approach**: Dual-Model ViT Ensemble + Test-Time Augmentation + Pixel-Level Forensics

---

## 🚨 Problem Statement

The rise of generative AI has created an unprecedented threat landscape:

- 📈 **96% increase** in deepfake content between 2023–2025
- 💸 **$12.5 Billion** in estimated fraud losses linked to synthetic media
- 🗳️ Deepfakes used in election misinformation, identity theft, and financial scams
- 🔍 Human eyes can correctly identify deepfakes only **53% of the time**

Existing solutions are either too expensive for individuals, too slow for real-time use, or lack explainability. **DeepGuard AI solves all three.**

---

## ✨ Features

| Feature | Description |
|---|---|
| � **Dual-Model Ensemble** | Two ViT models (99.3% + 92% accuracy) vote together with weighted averaging |
| 🔄 **Test-Time Augmentation** | 5 augmented variants per image (flip, rotate, sharpen) for robust predictions |
| 📐 **Multi-Scale Analysis** | Face analyzed at 3 crop margins (tight, medium, wide) for comprehensive coverage |
| 👤 **MTCNN Face Detection** | Automatic face isolation with adaptive weights when no face is found |
| 🔬 **Pixel-Level Forensics** | Real signal analysis: noise variance, edge consistency, color correlation, JPEG artifacts |
| 📊 **Calibrated Confidence** | Temperature-scaled softmax probabilities for honest uncertainty estimates |
| ⚡ **REST API** | FastAPI endpoint with file validation, structured logging, and rich JSON responses |
| 🌐 **Professional UI** | Streamlit dashboard with forensic metric cards, model agreement badges, quality indicators |
| 📁 **Multi-Format** | Supports JPG, JPEG, PNG, and WebP image formats |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                       │
│   Upload Image → Display Results → Forensic Metrics → Badges   │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP POST /detect
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND                          │
│         File Validation → Size Check → Type Whitelist           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION ENGINE                              │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │  MTCNN Face   │───▶│  Multi-Scale Crop (40, 80, 120 px)  │   │
│  │  Detection    │    └──────────────┬───────────────────────┘   │
│  └──────────────┘                   │                            │
│                                     ▼                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          Test-Time Augmentation (×5 per crop)             │   │
│  │   Original · H-Flip · Rotate CW · Rotate CCW · Sharpen   │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              DUAL-MODEL ENSEMBLE                         │    │
│  │                                                          │    │
│  │  Model A: dima806/deepfake_vs_real_image_detection       │    │
│  │           (99.3% accuracy, weight 60%)                   │    │
│  │                                                          │    │
│  │  Model B: prithivMLmods/Deep-Fake-Detector-v2-Model     │    │
│  │           (92% accuracy, weight 40%)                     │    │
│  │                                                          │    │
│  │  → Adaptive weights when no face detected (40/60)        │    │
│  │  → 30 total inferences averaged per image                │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           FORENSIC ANALYSIS MODULE                       │    │
│  │  Noise Variance · Edge Consistency · Color Correlation   │    │
│  │  JPEG Artifacts · Overall Suspicion Score                 │    │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 How It Works

### 1. Face Detection & Multi-Scale Cropping
MTCNN detects and isolates the face from the uploaded image. The face is cropped at **3 different margins** (40px, 80px, 120px) to capture both fine facial details and surrounding context.

### 2. Test-Time Augmentation (TTA)
Each crop is augmented into **5 variants**: original, horizontal flip, clockwise rotation (5°), counter-clockwise rotation (5°), and sharpened. This makes predictions robust against minor variations.

### 3. Dual-Model Ensemble
Two Vision Transformer models analyze every augmented crop independently:

| Model | Source | Accuracy | Default Weight |
|---|---|---|---|
| **Primary** | `dima806/deepfake_vs_real_image_detection` | 99.3% | 60% |
| **Secondary** | `prithivMLmods/Deep-Fake-Detector-v2-Model` | 92% | 40% |

Their softmax probabilities are averaged across all 30 passes, then merged with weighted voting.

### 4. Adaptive Weights
When MTCNN **cannot detect a face** (AI illustrations, anime, non-photo content), the weights automatically flip to **40/60**, giving more influence to the secondary model which better handles non-photographic AI content. The decision threshold also drops from 0.55 → 0.42.

### 5. Pixel-Level Forensics
Independent of the neural network, the system computes real signal-analysis metrics:
- **Noise Variance** — high-pass filter standard deviation
- **Edge Consistency** — Laplacian variance analysis
- **Color Correlation** — inter-channel R-G-B correlation
- **JPEG Artifacts** — 8×8 block boundary discontinuity

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/amanraj74/DeepGuard-AI.git
cd DeepGuard-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
pip install -r requirements.txt
```

### Run

```bash
# Terminal 1 — Start Backend (port 8000)
cd backend
python -m uvicorn main:app --reload --port 8000

# Terminal 2 — Start Frontend (port 8501)
cd frontend
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Note**: First run downloads ~660MB of model weights from HuggingFace. Subsequent runs use cached models.

---

## 📡 API Reference

### `GET /`
Health check and capability list.

### `GET /health`
Returns model status, device info, and label mappings.

### `POST /detect`
Upload an image for deepfake analysis.

**Request:**
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@photo.jpg"
```

**Response:**
```json
{
  "status": "success",
  "filename": "photo.jpg",
  "result": {
    "prediction": "Real",
    "confidence": 87.45,
    "face_detected": true,
    "scores": { "Real": 87.45, "Deepfake": 12.55 },
    "ensemble": {
      "models_used": 2,
      "agreement": true,
      "per_model": {
        "primary": { "model": "dima806/...", "prediction": "Real", "real": 91.2, "fake": 8.8 },
        "secondary": { "model": "prithivMLmods/...", "prediction": "Real", "real": 82.1, "fake": 17.9 }
      }
    },
    "forensics": {
      "noise_score": 42.3,
      "edge_score": 55.1,
      "color_score": 28.7,
      "jpeg_score": 15.2,
      "overall_suspicion": 35.8
    },
    "analysis_details": {
      "total_inferences": 30,
      "tta_passes": 5,
      "scales_analyzed": 3,
      "device": "cpu",
      "processing_time_ms": 18500
    }
  }
}
```

---

## 📂 Project Structure

```
DeepGuard-AI/
├── backend/
│   ├── main.py                      # FastAPI server + endpoints
│   ├── requirements.txt             # Backend dependencies
│   └── models/
│       ├── __init__.py
│       └── deepfake_detector.py     # Dual-model ensemble engine
├── frontend/
│   ├── app.py                       # Streamlit UI
│   └── requirements.txt             # Frontend dependencies
└── README.md
```

---

## �️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Models** | ViT (google/vit-base-patch16-224), HuggingFace Transformers |
| **Face Detection** | MTCNN (facenet-pytorch) |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | Streamlit |
| **Deep Learning** | PyTorch 2.2 |
| **Image Processing** | Pillow, NumPy |

---

## ⚠️ Limitations

- Models are trained on specific datasets and may not generalize to all deepfake types
- Performance may degrade on very low-resolution or heavily compressed images
- Designed for image classification — does not support video deepfake detection
- Processing takes 15–60 seconds per image on CPU (faster on GPU)

---

## 👤 Team Name - AI Core ( Member name - Aman Jaiswal )

Built for **IIT Bombay Hack & Break 2026** — Generative AI & Cybersecurity Innovation Challenge.

---

<div align="center">

**🛡️ DeepGuard AI** — Because seeing shouldn't always be believing.

</div>
