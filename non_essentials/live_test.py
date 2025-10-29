import cv2
from ultralytics import YOLO
import time

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "/home/testing/mobile_ocr/train/runs/m50_16v11m_v2_finetune_exp/weights/best.pt"
RTSP_URL = "rtsp://admin:Admin123@192.168.1.168:554/rtsp/streaming?channel=01&subtype=2"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# =========================
# LOAD MODEL
# =========================
print(f"📥 Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# =========================
# OPEN RTSP STREAM
# =========================
print(f"📹 Connecting to RTSP stream...")
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ Failed to open RTSP stream. Check camera IP, credentials, or network.")
    exit(1)

print("✅ Stream opened successfully!")

# =========================
# PROCESS STREAM
# =========================
prev_time = 0
fps_display_interval = 1  # seconds
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Stream disconnected or frame not received.")
        break

    # Run YOLOv11 inference
    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    # Draw boxes
    annotated_frame = results[0].plot()

    # Calculate FPS
    frame_count += 1
    current_time = time.time()
    if (current_time - prev_time) > fps_display_interval:
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0

    # Display FPS on frame
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Show window
    cv2.imshow("YOLOv11 RTSP Stream", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
print("🛑 Stream closed.")
