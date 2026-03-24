# ==============================
# AI BASED ADAS SYSTEM 
# This is part of our capstone project exploration
# ==============================

# Libraries used
import cv2
import numpy as np
import streamlit as st
import time
from ultralytics import YOLO

# ==============================
# Loading YOLO model (pretrained)
# Using nano version because it's faster for real-time
# ==============================
obj_model = YOLO('yolov8n.pt')

# ==============================
# Lane Detection Function
# Idea: detect only white & yellow lanes and ignore noise
# ==============================
def lane_detection_module(frame):

    # converting to HSV (better for color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # mask for white lanes
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # mask for yellow lanes
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # combine both masks
    combined = cv2.bitwise_or(white_mask, yellow_mask)

    # cleaning noise using morphology
    kernel = np.ones((5,5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # edge detection
    edges = cv2.Canny(combined, 50, 150)

    # Hough transform to get lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=100, maxLineGap=50)

    line_img = np.zeros_like(frame)

    # filtering only steep lines (lanes are usually not horizontal)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if abs(slope) > 0.5:
                cv2.line(line_img, (x1, y1), (x2, y2), (0,255,0), 5)

    # overlay on original frame
    return cv2.addWeighted(frame, 0.8, line_img, 1, 1)

# ==============================
# Object Detection Function
# Also doing simple collision logic
# ==============================
def object_detection_module(frame):

    results = obj_model(frame)[0]
    annotated = results.plot()

    warning = False

    # checking size of bounding boxes
    # bigger box = object closer
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        area = (x2 - x1) * (y2 - y1)

        if area > 60000:
            warning = True
            cv2.putText(annotated, "COLLISION WARNING", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    return annotated, warning

# ==============================
# Main processing pipeline
# combining both modules
# ==============================
def run_perception_pipeline(frame):

    frame, warning = object_detection_module(frame)
    frame = lane_detection_module(frame)

    return frame, warning

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="ADAS Dashboard", layout="wide")

st.title("🚗 AI ADAS Dashboard (Capstone Prototype)")

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload video", type=["mp4"])
run = st.sidebar.button("Start Simulation")

col1, col2 = st.columns([3,1])

video_display = col1.image([])
info_panel = col2.empty()

# ==============================
# Running the video
# ==============================
if uploaded and run:

    # saving uploaded file temporarily
    with open("temp.mp4", "wb") as f:
        f.write(uploaded.read())

    cap = cv2.VideoCapture("temp.mp4")

    prev_time = time.time()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960,540))

        processed, warning = run_perception_pipeline(frame)

        # calculating FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # displaying FPS
        cv2.putText(processed, f"FPS: {int(fps)}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        video_display.image(processed, channels="BGR")

        # updating side panel
        info_panel.markdown(f"""
        ### Live System Feed
        - FPS: {int(fps)}
        - Lane Detection: Running
        - Object Detection: YOLOv8
        - Collision Alert: {'YES' if warning else 'NO'}
        """)

    cap.release()

st.markdown("---")
st.markdown("### Notes")
st.markdown("- This is a simulation-based prototype")
st.markdown("- Hardware integration is planned in next phase")
st.markdown("- Built as part of capstone exploration")
