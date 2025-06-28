import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import tempfile
from datetime import datetime
import os
import pandas as pd
import time

@st.cache_resource
def load_model():
    return YOLO("models/yolov8n.pt")

def run_detection():
    st.subheader("Upload Image or Video for Detection")
    input_type = st.radio("Select Input Type", ["Image", "Video","Real-Time (Webcam)"])
    model = load_model()

    if input_type == "Image":
        image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if image_file:
            image = Image.open(image_file).convert("RGB")
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            results = model.predict(source=image_bgr)
            st.image(results[0].plot(), channels="BGR")

    elif input_type == "Video":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            frame_count = 0
            log = []
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 10 != 0:
                    continue
                results = model(frame)
                count = sum(int(cls) == 0 for cls in results[0].boxes.cls)
                log.append({"timestamp": datetime.now(), "person_count": count})
                stframe.image(results[0].plot(), channels="BGR")
            cap.release()
            df = pd.DataFrame(log)
            os.makedirs("analytics", exist_ok=True)
            df.to_csv("analytics/footfall.csv", index=False)
            st.success("Analytics saved.")

    
    elif input_type == "Real-Time (Webcam)":
        st.warning("Click 'Start Detection' to begin real-time detection using your webcam.")

        # Safe session state initialization
        if st.session_state.get("run") is None:
            st.session_state.run = False
        if st.session_state.get("log") is None:
            st.session_state.log = []

        # Buttons on separate rows
        if st.button("🟢 Start Detection"):
            st.session_state.run = True
            st.session_state.log = []  # reset logs

        if st.button("🔴 Stop Detection"):
            st.session_state.run = False

        if st.session_state.run:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            model.conf = 0.4
            last_logged = time.time()
            frame_interval = 1  # seconds

            while st.session_state.run and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.error("❌ Webcam not accessible.")
                    break

                results = model(frame)[0]
                annotated = results.plot()
                stframe.image(annotated, channels="BGR", use_container_width=True)

                # Log every 1 second
                if time.time() - last_logged >= frame_interval:
                    person_count = sum(int(cls) == 0 for cls in results.boxes.cls)
                    st.session_state.log.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "person_count": person_count
                    })
                    last_logged = time.time()

            cap.release()
            st.success("🛑 Webcam stopped.")

        # Save and update analytics when stopped
        if not st.session_state.run and st.session_state.log:
            df = pd.DataFrame(st.session_state.log)
            os.makedirs("analytics", exist_ok=True)
            df.to_csv("analytics/footfall.csv", index=False)
            st.session_state.log = []
            st.success("📊 Analytics saved. Dashboard updated.")
