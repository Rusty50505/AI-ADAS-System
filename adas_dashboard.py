import cv2
import numpy as np
import streamlit as st
import time
from ultralytics import YOLO

obj_model = YOLO('yolov8n.pt')


def lane_detection_module(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined = cv2.bitwise_or(white_mask, yellow_mask)
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(combined, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=100, maxLineGap=50)

    line_img = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if abs(slope) > 0.5:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return cv2.addWeighted(frame, 0.8, line_img, 1, 1)


def object_detection_module(frame):
    results = obj_model(frame)[0]
    annotated = results.plot()

    warning = False

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        area = (x2 - x1) * (y2 - y1)

        if area > 60000:
            warning = True
            cv2.putText(annotated, "COLLISION WARNING", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return annotated, warning


def process_video_frame(frame):
    frame, warning = object_detection_module(frame)
    frame = lane_detection_module(frame)
    return frame, warning


st.set_page_config(page_title="ADAS Dashboard", layout="wide")

st.title("🚗 AI ADAS Dashboard (Capstone Prototype)")

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload video", type=["mp4"])
run = st.sidebar.button("Start Simulation")

col1, col2 = st.columns([3, 1])

video_display = col1.image([])
info_panel = col2.empty()


if uploaded and run:
    with open("temp.mp4", "wb") as f:
        f.write(uploaded.read())

    cap = cv2.VideoCapture("temp.mp4")

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))

        processed, warning = process_video_frame(frame)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        cv2.putText(processed, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        video_display.image(processed, channels="BGR")

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
st.markdown("- Simulation-based prototype")
st.markdown("- Hardware integration planned next")
st.markdown("- Built during capstone exploration")
