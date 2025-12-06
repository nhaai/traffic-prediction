import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

# =====================================================
# LOAD CAMERA CONFIG
# =====================================================
with open("camera_config.json", "r") as f:
    CAM_CFG = json.load(f)

yolo = YOLO("yolov8s.pt")

# =====================================================
# NIGHT DETECTION
# =====================================================
def is_night(gray):
    brightness = np.mean(gray)
    return brightness < 90

def night_adjust(feats):
    feats["car"] *= 0.7
    feats["motorcycle"] *= 0.6
    feats["bus"] *= 0.85
    feats["truck"] *= 0.85
    feats["total"] *= 0.7
    feats["bbox_area_ratio"] *= 1.3
    feats["mean_bbox_area"] *= 1.15
    feats["max_bbox_area"] *= 1.10
    feats["edge_density"] *= 0.8
    feats["sharpness"] *= 1.1
    feats["is_night"] = 1

# =====================================================
# COMPUTE ZONAL FEATURES
# =====================================================
def compute_zones(h, boxes, cam_id):
    cam_zones = CAM_CFG[cam_id]["zones"]
    ztop = cam_zones["top"] # [y0%, y1%]
    zmid = cam_zones["mid"]
    zbot = cam_zones["bottom"]

    def in_zone(box, ratio_min, ratio_max):
        _, y1, _, y2 = box
        cy = (y1 + y2) / 2.0
        return (cy / h >= ratio_min) and (cy / h < ratio_max)

    zone_counts = {"top": 0, "mid": 0, "bottom": 0}
    for box in boxes:
        if in_zone(box, ztop[0], ztop[1]):
            zone_counts["top"] += 1
        elif in_zone(box, zmid[0], zmid[1]):
            zone_counts["mid"] += 1
        elif in_zone(box, zbot[0], zbot[1]):
            zone_counts["bottom"] += 1

    return zone_counts

# =====================================================
# RAIN DETECTION
# =====================================================
def detect_rain(gray):
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.sum(edges > 0) / gray.size
    brightness = np.mean(gray)
    if edge_density > 0.32 and brightness < 120:
        return 1
    return 0

# =====================================================
# VEHICLES DETECTION
# =====================================================
def detect_objects(img, night=False):
    if night:
        conf_car = 0.18
        conf_motor = 0.10
    else:
        conf_car = 0.30
        conf_motor = 0.22

    base_conf = min(conf_car, conf_motor)
    results = yolo(img, conf=base_conf, verbose=False)[0]

    vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    objs = []

    for box in results.boxes:
        cls = int(box.cls)
        if cls not in vehicle_classes:
            continue

        label = vehicle_classes[cls]
        conf = float(box.conf)
        if label == "motorcycle" and conf < conf_motor:
            continue
        if label == "car" and conf < conf_car:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if x2 <= x1 or y2 <= y1:
            continue

        objs.append({
            "label": label,
            "conf": conf,
            "bbox": (x1, y1, x2, y2)
        })

    objs = sorted(objs, key=lambda o: o["conf"], reverse=True)
    return suppress_duplicates(objs)

# =====================================================
# NMS SECONDARY
# =====================================================
def suppress_duplicates(objs, iou_thres=0.6):
    out = []
    for o in objs:
        keep = True
        for ex in out:
            if IOU(o["bbox"], ex["bbox"]) > iou_thres:
                keep = False
                break
        if keep:
            out.append(o)
    return out

def IOU(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)

    return inter_area / float(a_area + b_area - inter_area)

# =====================================================
# FEATURES DETECTION
# =====================================================
def extract_features(img, cam_id):
    h, w, _ = img.shape
    area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    night = is_night(gray)
    rain = detect_rain(gray)
    objs = detect_objects(img, night)

    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    bbox_areas = []
    box_list = []

    cam_zones = CAM_CFG[cam_id]["zones"]
    zmid = cam_zones["mid"]
    zbot = cam_zones["bottom"]

    bottom_motor = 0
    mid_car = 0

    for o in objs:
        lbl = o["label"]
        x1, y1, x2, y2 = o["bbox"]
        bbox_area = (x2 - x1) * (y2 - y1)
        ratio = bbox_area / area
        if ratio < 0.001:
            continue

        counts[lbl] += 1
        bbox_areas.append(bbox_area)
        box_list.append((x1, y1, x2, y2))

        cy = (y1 + y2) / 2.0
        ry = cy / h

        if zbot[0] <= ry < zbot[1] and lbl == "motorcycle":
            bottom_motor += 1
        if zmid[0] <= ry < zmid[1] and lbl == "car":
            mid_car += 1

    if night and counts["motorcycle"] < 5:
        counts["motorcycle"] = int(counts["motorcycle"] * 1.25)

    # total vehicles
    total = sum(counts.values())

    # grayscale image
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 70, 140)
    edge_density = float(np.sum(edges > 0) / area)

    # bbox statistics
    if bbox_areas:
        bbox_area_ratio = sum(bbox_areas) / area
        mean_bbox_area = float(np.mean(bbox_areas))
        max_bbox_area = max(bbox_areas)
    else:
        bbox_area_ratio = mean_bbox_area = max_bbox_area = 0.0

    # zone-based counts
    zone_counts = compute_zones(h, box_list, cam_id)

    feats = {
        "car": counts["car"],
        "motorcycle": counts["motorcycle"],
        "bus": counts["bus"],
        "truck": counts["truck"],
        "total": total,

        "bbox_area_ratio": bbox_area_ratio,
        "mean_bbox_area": mean_bbox_area,
        "max_bbox_area": max_bbox_area,
        "brightness": brightness,
        "sharpness": sharpness,
        "edge_density": edge_density,

        "zone_top": zone_counts["top"],
        "zone_mid": zone_counts["mid"],
        "zone_bottom": zone_counts["bottom"],

        "bottom_motor": bottom_motor,
        "mid_car": mid_car,
        "cluster_density": bbox_area_ratio,

        "is_night": int(night),
        "is_rain": int(rain)
    }

    # night-mode correction
    if night:
        night_adjust(feats)

    return feats

# =====================================================
# CAM ID EXTRACTION
# =====================================================
def extract_cam_id(filename):
    name = os.path.basename(filename).lower()
    if name.startswith("cam") and len(name) >= 5:
        return name[:5] # cam09, cam10 ...
    return None
