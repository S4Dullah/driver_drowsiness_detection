# 🚗 Driver Drowsiness Detection — Raspberry Pi 4

Real-time driver drowsiness detection system running on Raspberry Pi 4. Detects fatigue using eye blink rate (EAR), head pose estimation and PERCLOS analysis via MediaPipe, with a TFLite model for eye state classification.

---

## 📋 Features

- **EAR (Eye Aspect Ratio)** — detects eye closure via facial landmarks
- **PERCLOS** — measures percentage of eye closure over a 10-second window
- **Head Pose Estimation** — detects head drop (pitch), yaw and roll via MediaPipe
- **TFLite AI Model** — classifies eye state (open/closed) from cropped eye images
- **Yawn Detection** — detects yawning via MAR (Mouth Aspect Ratio)
- **Automatic Calibration** — 15-second baseline calibration at startup
- **GPIO Output** — Red / Yellow / Green LEDs + Buzzer alerts
- **MJPEG Stream** — live video feed accessible at `http://<pi-ip>:8080`
- **Glare Handling** — dynamic gamma correction and glare masking for low-light conditions

---

## 🔧 Hardware Requirements

| Component | Details |
|---|---|
| Raspberry Pi 4 | 2 GB RAM or more recommended |
| Camera | Raspberry Pi Camera Module (libcamera) |
| LEDs | Red (GPIO 17), Green (GPIO 27), Yellow (GPIO 22) |
| Buzzer | GPIO 18 |
| Power | 5V 3A USB-C |

---

## 📁 Project Structure

```
yorgunluk_tespiti_pi/
├── src/
│   ├── main.py             # Main application
│   ├── ear_hesap.py        # EAR / MAR calculation
│   ├── headpose.py         # Head pose estimation
│   ├── perclos.py          # PERCLOS calculation
│   ├── kalibrasyon.py      # Calibration logic
│   └── camera_server.py    # picamera2 subprocess server
├── model/
│   └── model.tflite        # TFLite eye state classifier
├── setup_pi.sh             # Installation script
├── requirements.txt        # Python dependencies
└── yorgunluk.service       # systemd service file
```

---

## ⚙️ Installation

### Prerequisites

- Raspberry Pi 4 with Raspberry Pi OS (64-bit)
- [Miniforge3 for aarch64](https://github.com/conda-forge/miniforge) installed
- Conda environment named `proje_ortami` with **Python 3.10**

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/S4Dullah/yorgunluk_tespiti_pi.git
cd yorgunluk_tespiti_pi

# 2. Create conda environment
conda create -n proje_ortami python=3.10
conda activate proje_ortami

# 3. Run the setup script
bash setup_pi.sh
```

The setup script handles:
- System packages (`lgpio`, `libportaudio2`, `libatlas-base-dev`, etc.)
- OpenCV conflict resolution
- Python dependencies from `requirements.txt`
- Installation verification

---

## ▶️ Usage

```bash
conda activate proje_ortami
python src/main.py
```

### Live video stream

Open a browser and navigate to:
```
http://<raspberry-pi-ip>:8080
```

### Run as a system service (autostart on boot)

```bash
sudo cp yorgunluk.service /etc/systemd/system/
sudo systemctl enable yorgunluk.service
sudo systemctl start yorgunluk.service
```

---

## 🚦 Alert System

| LED State | Meaning |
|---|---|
| 🟢 Green solid | Driver is alert — normal |
| 🟡 Yellow solid | No face detected or head turned |
| 🟡 Yellow + Green | EAR unreliable (head angle too high) |
| 🔴 Red solid | Drowsiness detected (PERCLOS threshold exceeded) |
| 🔴 Red blinking | Critical — head drop detected |

Buzzer beeps slowly on drowsiness detection, continuously on head drop.

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| mediapipe | 0.10.9 | Face landmark detection |
| opencv-contrib-python | 4.11.0.86 | Image processing |
| tflite-runtime | 2.13.0 | AI inference (aarch64) |
| numpy | 1.24.3 | Numerical operations |
| gpiozero | ≥2.0 | GPIO control |

> **Note:** `requirements.txt` targets **Python 3.10 on ARM64 (Raspberry Pi 4)**. These packages are not compatible with Python 3.13 or x86 systems.

---

## 📷 Camera Architecture

The camera uses `picamera2` (libcamera) which requires the system Python (`/usr/bin/python3`). A subprocess server (`camera_server.py`) streams raw frames to the main application via stdout, allowing the conda environment to consume frames without libcamera compatibility issues.

---

## 📄 License

MIT License
