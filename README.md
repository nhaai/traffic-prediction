# 🚦 Traffic Prediction

This project provides tools for:

1. **Capturing traffic camera frames** from public HCMC traffic streams  
2. **Cleaning & resizing images**
3. **Detecting vehicles using YOLOv8s**
4. **Extracting features**
5. **Auto-labeling images into free_flow / moderate / congested**
6. **Splitting data into train/val/test**
7. **Training a decision tree model**

---

## 📌 1. System Requirements

### Ubuntu 24.04 LTS on WSL
Install required system libraries:

```bash
sudo apt install -y libnss3 libasound2t64
```

### Node.js
Tested on Node v18.13.0

### Python
Tested on Python v3.9.2

### Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 2. Project Structure

```bash
project/
│
├── capture_cam.js           # Capture script for traffic cameras
│
├── dataset_raw/             # Raw captured images
├── dataset_cleaned/         # Cleaned + resized images
├── dataset_split/           # train/val/test output folders
│
├── prepare_dataset.py       # Cleaning, resizing, YOLO detection, feature extraction, auto labeling, splitting
├── train_model.py           # Train decision tree model
├── demo_predict.py          # Single-image prediction using YOLO + ML
├── draw_tree.py             # Export decision tree visualization
│
├── app.py                   # Flask + Tailwind demo web interface
├── static/
│   ├── uploads/             # Uploaded images
│   └── decision_tree.png    # Generated ML tree graph
│
├── templates/
│   └── index.html           # Tailwind UI
|
├── requirements.txt
├── Dockerfile
├── environment.yml
└── README.md
```
---

## 🎥 3. Capturing Camera Streams

Use the Node script **capture_cam.js** to capture images/frames from public traffic camera streams in Ho Chi Minh City.

Each command specifies:
- `--cam_id` → ID used for saving or naming files
- `--url` → the full camera page containing the embedded HLS stream (m3u8)

```bash
node capture_cam.js --cam_id CAM_ID --url "FULL_CAMERA_URL"
```

Examples:

```bash
node capture_cam.js --cam_id cam07 \
  --url "https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5a8254f25058170011f6eac5&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8"
node capture_cam.js --cam_id cam08 \
  --url "https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=56df8159c062921100c143dc&camMode=camera&videoUrl=http://125.234.114.126:11984/api/stream.m3u8?src=N%C3%BAt%20giao%20Th%E1%BB%A7%20%C4%90%E1%BB%A9c%201&mp4"
node capture_cam.js --cam_id cam09 \
  --url "https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=58b5510817139d0010f35d4e&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8"
node capture_cam.js --cam_id cam10 \
  --url "https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=5a6069238576340017d0661c&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8"
node capture_cam.js --cam_id cam11 \
  --url "https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=58ad69c4bd82540010390be7&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8"
```

---

## 🧹 3. Dataset Processing

Run:

```bash
python prepare_dataset.py
```

This performs:

- Cleaning (Gaussian blur)
- Resizing to 224×224
- YOLOv8s vehicle detection
- Feature extraction
- Auto labeling (free_flow/moderate/congested)
- train/val/test split (70/20/10)

Outputs stored in:

```
dataset_cleaned/
dataset_split/
├── train/
├── val/
└── test/
dataset_features.csv
```

---

## 🤖 4. Training model

Run:

```bash
python train_model.py
```

Saves model to:

```
model.pkl
```

---

## 🌐 5. Demo

Run:

```bash
python app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

---

## 📝 Notes

- Some traffic camera URLs rotate or expire; refresh links if capture fails.
- Ensure WSL has video dependencies installed; Chromium may require additional codecs depending on stream type.
- You may run multiple camera capture processes in parallel.
