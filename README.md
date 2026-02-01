# 🔍 Face Recognition with ArcFace (ONNX) & 5-Point Alignment

A **CPU-only, research-grade face recognition system** built with **ArcFace embeddings** and **5-point facial landmark alignment**, designed for **clarity, robustness, and real-world deployment** on machines without GPU acceleration.


---

## ✨ Features

- ⚙️ **CPU-Only Inference** – runs smoothly on laptops and low-resource machines  
- 🧠 **ArcFace (ONNX, ResNet-50)** – 512-dimensional L2-normalized embeddings  
- 📐 **5-Point Face Alignment** – similarity transform to canonical 112×112 faces  
- 🎥 **Real-Time Recognition** – multi-face detection with temporal smoothing  
- 🔓 **Open-Set Recognition** – automatically rejects unknown identities  
- 📊 **Threshold Evaluation** – FAR / FRR based decision tuning  
- 🧩 **Modular Pipeline** – each stage testable independently  
- 🔒 **Face Locking** – lock onto a specific identity and track actions (movement, blinks, smiles)  

---

## 🖥️ System Requirements

| Component | Requirement |
|---------|-------------|
| Python | 3.9+ (tested on 3.11) |
| OS | Windows / macOS / Linux |
| Camera | Webcam |
| RAM | ≥ 2 GB |

Check Python version:

```bash
python --version
1️⃣ Clone Repository
git clone https://github.com/Nik-ta07/Face-Locking.git
cd -Face-Recog-arc-onnx

2️⃣ Create Virtual Environment
python3.11 -m venv .venv


Activate:

Windows (PowerShell)

.venv\Scripts\Activate.ps1


macOS / Linux

source .venv/bin/activate

3️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

🧠 ArcFace Model Setup

Download the official InsightFace ArcFace ONNX model:

curl -L -o buffalo_l.zip \
https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download

unzip buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm buffalo_l.zip *.onnx

📁 Project Structure
Face-Recog-arc-onnx/
│
├── src/
│   ├── camera.py        # Camera validation
│   ├── detect.py        # Haar face detection
│   ├── landmarks.py     # 5-point landmark extraction
│   ├── align.py         # 112×112 face alignment
│   ├── embed.py         # ArcFace embedding extraction
│   ├── enroll.py        # Identity enrollment
│   ├── evaluate.py      # FAR / FRR threshold evaluation
│   ├── recognize.py    # Live face recognition
│   └── lock.py          # Face locking and action tracking
│
├── data/
│   ├── enroll/          # Aligned enrollment images
│   ├── db/              # Face database (NPZ + JSON)
│   └── history/         # Action history files
│
├── models/
│   └── embedder_arcface.onnx
│
├── requirements.txt
└── README.md

🚀 Quick Start

Test each module independently:

python -m src.camera
python -m src.detect
python -m src.landmarks
python -m src.align
python -m src.embed


Enroll identities and start recognition:

python -m src.enroll
python -m src.evaluate
python -m src.recognize
python -m src.lock

🔄 Pipeline Overview
Enrollment Pipeline
Camera
 → Face Detection
 → 5-Point Landmarks
 → Alignment (112×112)
 → ArcFace Embedding
 → L2 Normalization
 → Mean Template
 → Database Storage

Recognition Pipeline
Camera
 → Detection + Alignment
 → ArcFace Embedding
 → Cosine Distance Matching
 → Threshold Decision
 → Identity / Unknown

---

## 🔒 Face Locking Feature

The Face Locking feature extends the recognition system to track a specific enrolled identity and detect their actions over time.

### Features

- **Manual Face Selection** – Choose which enrolled identity to lock onto
- **Automatic Locking** – System automatically locks when target face is detected with high confidence
- **Stable Tracking** – Continues tracking even with brief recognition failures (2-second timeout)
- **Action Detection**:
  - **Face Movement** – Detects left/right movement
  - **Eye Blinks** – Detects eye blinks using Eye Aspect Ratio (EAR)
  - **Smiles/Laughs** – Detects smiles using Mouth Aspect Ratio (MAR)
- **Action History** – Records all detected actions to timestamped files

### Usage

1. **Enroll faces first** (if not already done):
   ```bash
   python -m src.enroll
   ```

2. **Start face locking**:
   ```bash
   python -m src.lock
   ```

3. **Select target face** from the list of enrolled identities

4. **Controls**:
   - `q` – Quit
   - `l` – Manually lock/unlock
   - `r` – Reload database
   - `+/-` – Adjust recognition threshold

### How It Works

1. **Face Selection**: When you start the system, you select which enrolled identity to track
2. **Auto-Locking**: When the target face appears and is recognized with high confidence (>0.7 similarity), the system automatically locks onto it
3. **Tracking**: Once locked, the system tracks the face position and detects:
   - **Movement**: Calculates face center position changes to detect left/right movement
   - **Blinks**: Uses MediaPipe FaceMesh to calculate Eye Aspect Ratio (EAR) and detects when eyes close
   - **Smiles**: Uses Mouth Aspect Ratio (MAR) to detect when mouth opens wider (smile/laugh)
4. **History Recording**: All detected actions are recorded to a file with format:
   ```
   <face_name>_history_<timestamp>.txt
   ```
   Example: `aline_history_20260129112099.txt`

### Action History File Format

Each action is recorded with:
- **Timestamp** – When the action occurred
- **Action Type** – Type of action (movement, blink, smile)
- **Description** – Human-readable description
- **Value** – Optional numerical value (distance, ratio, etc.)

Example:
```
Face Locking History for: Aline
Started: 2026-01-29 11:20:45
------------------------------------------------------------

2026-01-29 11:20:50 | movement   | Face moved left (value: 35.234)
2026-01-29 11:20:52 | blink      | Eye blink detected (value: 0.189)
2026-01-29 11:20:55 | smile      | Smile or laugh detected (value: 0.623)
2026-01-29 11:20:58 | movement   | Face moved right (value: 42.156)
```

### Lock Behavior

- **Lock Acquisition**: Automatically locks when target face is detected with similarity > 0.7
- **Lock Maintenance**: Stays locked even if recognition briefly fails
- **Lock Release**: Releases lock if target face is not seen for 2 seconds
- **Manual Control**: Press `l` to manually lock/unlock at any time

### Notes

- The system tracks only the selected identity, ignoring other faces
- Action detection uses MediaPipe FaceMesh for accurate landmark detection
- History files are saved in `data/history/` directory
- The system is CPU-only and runs in real-time

Face Locking Pipeline
Camera
 → Face Detection
 → Identity Recognition
 → Lock onto Target Face
 → Action Detection (movement, blinks, smiles)
 → Action History Recording
 