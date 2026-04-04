import cv2
import numpy as np
import streamlit as st
import time
import tempfile
import os
from ultralytics import YOLO

obj_model = YOLO('yolov8n.pt')

def lane_detection_module(frame):
    h, w = frame.shape[:2]
    roi_vertices = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.45), int(h * 0.55)),
        (int(w * 0.55), int(h * 0.55)),
        (int(w * 0.95), h)
    ]], dtype=np.int32)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
    combined = cv2.bitwise_or(white_mask, yellow_mask)

    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    roi_mask = np.zeros_like(combined)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    masked = cv2.bitwise_and(combined, roi_mask)

    edges = cv2.Canny(masked, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=80, maxLineGap=60)

    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) < 0.4:
                continue
            if slope < 0:
                left_lines.append(line[0])
            else:
                right_lines.append(line[0])

    def average_lane(lane_lines, frame_h):
        if not lane_lines:
            return None
        x_coords = np.array([[l[0], l[2]] for l in lane_lines]).flatten()
        y_coords = np.array([[l[1], l[3]] for l in lane_lines]).flatten()
        if len(set(x_coords)) < 2:
            return None
        fit = np.polyfit(y_coords, x_coords, 1)
        y1_pt = frame_h
        y2_pt = int(frame_h * 0.58)
        x1_pt = int(np.polyval(fit, y1_pt))
        x2_pt = int(np.polyval(fit, y2_pt))
        return (x1_pt, y1_pt, x2_pt, y2_pt)

    line_img = np.zeros_like(frame)
    left_lane = average_lane(left_lines, h)
    right_lane = average_lane(right_lines, h)

    for lane in [left_lane, right_lane]:
        if lane:
            cv2.line(line_img, (lane[0], lane[1]), (lane[2], lane[3]), (0, 255, 0), 6)

    if left_lane and right_lane:
        pts = np.array([[
            (left_lane[0], left_lane[1]),
            (left_lane[2], left_lane[3]),
            (right_lane[2], right_lane[3]),
            (right_lane[0], right_lane[1])
        ]], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, pts, (0, 200, 0))
        frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

    return cv2.addWeighted(frame, 0.9, line_img, 1, 0)


def object_detection_module(frame):
    results = obj_model(frame, verbose=False)[0]
    annotated = results.plot()
    warning = False
    critical_classes = {0, 1, 2, 3, 5, 7}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0]
        area = (x2 - x1) * (y2 - y1)
        if area > 50000 and cls_id in critical_classes:
            warning = True
            cv2.putText(annotated, "⚠ COLLISION WARNING", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            break

    return annotated, warning


def process_video_frame(frame):
    frame, warning = object_detection_module(frame)
    frame = lane_detection_module(frame)
    return frame, warning


st.set_page_config(page_title="ADAS Dashboard", layout="wide", page_icon="🚗")
st.title("🚗 AI ADAS Dashboard — Capstone Prototype")
st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader("Upload driving video", type=["mp4", "mov", "avi"])
run = st.sidebar.button("▶ Start Simulation")
skip_frames = st.sidebar.slider("Frame skip (speed vs quality)", 0, 5, 1)

col1, col2 = st.columns([3, 1])
video_display = col1.empty()
info_panel = col2.empty()

if uploaded and run:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    prev_time = time.time()
    frame_count = 0
    warning_count = 0

    stop = st.sidebar.button("⏹ Stop")

    while cap.isOpened():
        if stop:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % (skip_frames + 1) != 0:
            continue

        frame = cv2.resize(frame, (960, 540))
        processed, warning = process_video_frame(frame)

        if warning:
            warning_count += 1

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(processed, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        video_display.image(processed, channels="BGR", use_container_width=True)

        info_panel.markdown(f"""
        ### 📊 Live System Feed
        | Metric | Status |
        |---|---|
        | FPS | {int(fps)} |
        | Frame | {frame_count} |
        | Lane Detection | ✅ Active |
        | Object Model | YOLOv8n |
        | Collision Alert | {'🔴 YES' if warning else '🟢 NO'} |
        | Total Warnings | {warning_count} |
        """)

    cap.release()
    os.unlink(tfile.name)
    st.success("Simulation complete.")

st.markdown("---")
st.markdown("**Prototype Notes:** Simulation-based · Hardware integration planned · Built during capstone exploration")
