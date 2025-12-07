# 🚦 Traffic Prediction

This project provides tools for:

1. **Capturing traffic camera frames** from public HCMC traffic streams  
2. **Cleaning & resizing images**
3. **Extracting features**
4. **Labeling images**
5. **Splitting data into train/val/test**
6. **Training a decision tree model**

---

## 📌 1. System Requirements

### Ubuntu 24.04 LTS on WSL
Install required system libraries:

```bash
sudo apt install -y libnss3 libasound2t64
```

### Python, Node.js
Tested on Python v3.9.2, Node v18.13.0

### Python Dependencies
```bash
python3.9 -m venv .venv39
source .venv39/bin/activate
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
├── crowd_counter/
│   ├── csrnet.py            # CSRNet wrapper for crowd_density feature
│   └── model.pth            # Pretrained CSRNet weights (ShanghaiTech Part A)
│                            # https://drive.google.com/file/d/1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH/view
│
├── prepare_dataset.py       # Cleaning, resizing, detection, feature extraction, auto labeling, splitting
├── train_model.py           # Train decision tree model
├── draw_tree.py             # Export decision tree visualization
│
├── console.py               # Run traffic prediction from the terminal
├── app.py                   # Flask + Tailwind demo web interface
├── static/
│   └── uploads/             # Uploaded images
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
You may run multiple camera capture processes in parallel.
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
python3 prepare_dataset.py --reset --all
```

This performs:

- Cleaning (Gaussian blur)
- Resizing to 224×224
- Feature extraction
- Labeling (free_flow/moderate/congested)
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

During feature extraction, if `crowd_counter/model.pth` is available,
the script also computes a `crowd_density` feature using CSRNet
(crowd counting). If the model is missing, `crowd_density` is set to 0
and the rest of the pipeline still works.

---

## 🤖 4. Training model

Run:

```bash
python3 train_model.py
```

Saves model to:

```
model.pkl
```

---

## 🌐 5. Demo

Run:

```bash
python3 app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

Or:

```bash
python3 console.py static/uploads/congested/cam07_20251203_173405.png
```

---

## 📊 6. Feature Set Used

| Feature           | Role                                                                     |
|-------------------|--------------------------------------------------------------------------|
| car               | Number of cars — strong density indicator                                |
| motorcycle        | Dominant vehicle type in Vietnam → highly impactful                      |
| bus               | Helps detect heavy traffic flow                                          |
| truck             | Large vehicles often contribute to slowdowns                             |
| total             | Overall vehicle count                                                    |
| bbox_area_ratio   | Percentage of image area occupied by vehicles — strong congestion signal |
| mean_bbox_area    | Larger bounding boxes → vehicles closer to camera → possible bottleneck  |
| max_bbox_area     | Detects very large/close vehicles blocking the camera view               |
| brightness        | Related to day/night conditions                                          |
| sharpness         | Indicates blur/noise in the image affecting detection quality            |
| edge_density      | Texture density — correlates with object/vehicle density                 |
| zone_top          | Vehicle count in the far region of the camera view                       |
| zone_mid          | Vehicle count in the mid-region — affects flow state                     |
| zone_bottom       | Vehicles closest to the camera — strong congestion indicator             |
| bottom_motor      | Motorcycles in the bottom zone — early sign of traffic jam               |
| mid_car           | Cars in the mid-zone — relates to slowing traffic                        |
| cluster_density   | Bounding-box density — captures clustering of vehicles                   |
| is_night          | Night flag used for adaptive feature adjustment                          |
| is_rain           | Rain flag (edge + brightness) impacting visibility and flow              |
| crowd_density     | CSRNet-based density estimate, robust signal for free/moderate/congested |

---

## 📝 Notes

- Conduct data collection and capture at least 2,000 samples (with supporting evidence).
- Then proceed with labeling (use LabelImg to crop and label image data), and perform data preprocessing and feature extraction for other types of data.
- The first evaluation session includes: the dataset, data preprocessing and feature extraction, train/validation/test split, running one machine learning model (Decision Tree), and reporting the accuracy metrics: Recall, F1, and Precision.
- Report file: must follow a master's thesis structure, include all required sections, with 30–40 pages of main content.
