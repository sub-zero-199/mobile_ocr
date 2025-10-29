# import cv2
# from ultralytics import YOLO
# import time

# # =========================
# # CONFIGURATION
# # =========================
# # --- Model 1 Configuration ---
# MODEL_PATH_1 = "/home/testing/mobile_ocr/train/runs/m50_16v11m_v2_finetune_exp/weights/best.pt"
# CONF_THRESHOLD_1 = 0.25
# IOU_THRESHOLD_1 = 0.45

# # --- Model 2 Configuration (Example) ---
# # NOTE: Replace this with your actual second model's path and parameters
# MODEL_PATH_2 = "/home/testing/mobile_ocr/bestsh.pt" 
# CONF_THRESHOLD_2 = 0.30
# IOU_THRESHOLD_2 = 0.50

# # --- Stream Configuration ---
# RTSP_URL = "rtsp://admin:Admin123@192.168.1.168:554/rtsp/streaming?channel=01&subtype=2"
# IMG_SIZE = 640 # Image size for both models

# # =========================
# # LOAD MODELS
# # =========================
# print(f"📥 Loading Model 1 from: {MODEL_PATH_1}")
# model1 = YOLO(MODEL_PATH_1)

# print(f"📥 Loading Model 2 from: {MODEL_PATH_2}")
# try:
#     model2 = YOLO(MODEL_PATH_2)
# except Exception as e:
#     print(f"❌ Failed to load Model 2. Check path/file: {e}")
#     # Consider exiting or setting model2 to None if you want the script to continue with one model
#     exit(1)


# # =========================
# # OPEN RTSP STREAM
# # =========================
# print(f"📹 Connecting to RTSP stream...")
# cap = cv2.VideoCapture(RTSP_URL)

# if not cap.isOpened():
#     print("❌ Failed to open RTSP stream. Check camera IP, credentials, or network.")
#     exit(1)

# print("✅ Stream opened successfully!")

# # =========================
# # PROCESS STREAM
# # =========================
# prev_time = 0
# fps_display_interval = 1  # seconds
# frame_count = 0
# fps = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("⚠️ Stream disconnected or frame not received.")
#         break

#     # ---------------------------------
#     # 1. Run Model 1 Inference (e.g., OCR detection)
#     # ---------------------------------
#     results1 = model1.predict(
#         source=frame,
#         imgsz=IMG_SIZE,
#         conf=CONF_THRESHOLD_1,
#         iou=IOU_THRESHOLD_1,
#         verbose=False
#     )
    
#     # Start with the original frame for plotting
#     annotated_frame = frame.copy() 
    
#     # Draw boxes for Model 1 (plots on the copy of the frame)
#     # NOTE: Using results[0].plot() is an easy way, but you can iterate results1[0].boxes to customize colors/labels
#     annotated_frame = results1[0].plot(
#         img=annotated_frame, 
#         labels=True, 
#         conf=True, 
#         line_width=2,
#         # Customize color for Model 1 detections
#         # boxes=[(0, 0, 255)], # Uncomment and specify a color if needed
#         ) 

#     # ---------------------------------
#     # 2. Run Model 2 Inference (e.g., Vehicle detection)
#     # ---------------------------------
#     results2 = model2.predict(
#         source=annotated_frame, # Use the frame already annotated by Model 1
#         imgsz=IMG_SIZE,
#         conf=CONF_THRESHOLD_2,
#         iou=IOU_THRESHOLD_2,
#         verbose=False
#     )

#     # Draw boxes for Model 2 (plots on the already annotated frame)
#     annotated_frame = results2[0].plot(
#         img=annotated_frame, 
#         labels=True, 
#         conf=True, 
#         line_width=2,
#         # Customize color for Model 2 detections
#         # boxes=[(255, 0, 0)], # Uncomment and specify a different color
#         ) 
    
#     # ---------------------------------
#     # 3. FPS and Display
#     # ---------------------------------

#     # Calculate FPS
#     frame_count += 1
#     current_time = time.time()
#     if (current_time - prev_time) > fps_display_interval:
#         # FPS is based on the total processing time for both models
#         fps = frame_count / (current_time - prev_time)
#         prev_time = current_time
#         frame_count = 0

#     # Display FPS on frame
#     cv2.putText(
#         annotated_frame,
#         f"FPS: {fps:.2f} (2 Models)", # Indicate 2 models are running
#         (10, 30),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 0),
#         2,
#         cv2.LINE_AA,
#     )

#     # Show window
#     cv2.imshow("Multi-Model YOLO RTSP Stream", annotated_frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # =========================
# # CLEANUP
# # =========================
# cap.release()
# cv2.destroyAllWindows()
# print("🛑 Stream closed.")

import cv2
from ultralytics import YOLO
import time
import torch # Import PyTorch for device check

# =========================
# DEVICE CONFIGURATION
# =========================
# Check for GPU and select device
if torch.cuda.is_available():
    DEVICE = 'cuda:0'  # Use the first GPU
    DEVICE_STATUS = "GPU (CUDA)"
    # Initialize PyTorch for the first time on GPU to reduce initial latency
    torch.cuda.init() 
else:
    DEVICE = 'cpu'
    DEVICE_STATUS = "CPU"

print(f"🖥️ Selected Device: {DEVICE_STATUS} ({DEVICE})")

# =========================
# CONFIGURATION
# =========================
# --- Model 1 Configuration ---
MODEL_PATH_1 = "/home/testing/mobile_ocr/train/runs/m50_16v11m_v2_finetune_exp/weights/best.pt"
CONF_THRESHOLD_1 = 0.25
IOU_THRESHOLD_1 = 0.45

# --- Model 2 Configuration (Example) ---
MODEL_PATH_2 = "/home/testing/mobile_ocr/bestsh.pt" 
CONF_THRESHOLD_2 = 0.30
IOU_THRESHOLD_2 = 0.50

# --- Stream Configuration ---
RTSP_URL = "rtsp://admin:Admin123@192.168.1.168:554/rtsp/streaming?channel=01&subtype=2"
IMG_SIZE = 640 # Image size for both models

# =========================
# LOAD MODELS
# =========================
print(f"📥 Loading Model 1 from: {MODEL_PATH_1}")
# Pass the determined device when initializing the model
model1 = YOLO(MODEL_PATH_1) 
model1.to(DEVICE) # Ensure the model weights are on the selected device

print(f"📥 Loading Model 2 from: {MODEL_PATH_2}")
try:
    model2 = YOLO(MODEL_PATH_2)
    model2.to(DEVICE) # Ensure the model weights are on the selected device
except Exception as e:
    print(f"❌ Failed to load Model 2. Check path/file: {e}")
    exit(1)

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

    # ---------------------------------
    # 1. Run Model 1 Inference 
    # ---------------------------------
    results1 = model1.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD_1,
        iou=IOU_THRESHOLD_1,
        verbose=False,
        device=DEVICE # Explicitly use the determined device
    )
    
    # Start with the original frame for plotting
    annotated_frame = frame.copy() 
    
    # Draw boxes for Model 1
    annotated_frame = results1[0].plot(
        img=annotated_frame, 
        labels=True, 
        conf=True, 
        line_width=2,
        ) 

    # ---------------------------------
    # 2. Run Model 2 Inference 
    # ---------------------------------
    results2 = model2.predict(
        source=annotated_frame, # Use the frame already annotated by Model 1
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD_2,
        iou=IOU_THRESHOLD_2,
        verbose=False,
        device=DEVICE # Explicitly use the determined device
    )

    # Draw boxes for Model 2
    annotated_frame = results2[0].plot(
        img=annotated_frame, 
        labels=True, 
        conf=True, 
        line_width=2,
        ) 
    
    # ---------------------------------
    # 3. FPS and Display
    # ---------------------------------

    # Calculate FPS
    frame_count += 1
    current_time = time.time()
    if (current_time - prev_time) > fps_display_interval:
        fps = frame_count / (current_time - prev_time)
        prev_time = current_time
        frame_count = 0

    # Display FPS and Device Status on frame
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
    cv2.putText(
        annotated_frame,
        f"Device: {DEVICE_STATUS}", # Display the determined device
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255), # Yellow color for device status
        2,
        cv2.LINE_AA,
    )


    # Show window
    cv2.imshow("Multi-Model YOLO RTSP Stream", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
print("🛑 Stream closed.")