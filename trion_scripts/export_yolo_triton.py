from ultralytics import YOLO
import shutil
import os

# Configuration - Update these paths
MODEL_PATH = "/home/testing/mobile_ocr/final.pt"
EXPORT_DESTINATION = "/home/testing/mobile_ocr/trion_repo/yolov11m/1/model.onnx"

# Load your trained model
model = YOLO(MODEL_PATH)

# Export with opset 17 for maximum compatibility
export_path = model.export(
    format="onnx",
    opset=17,
    simplify=True,
    dynamic=False,
    imgsz=640
)

print(f"Model exported successfully with opset 17!")
print(f"Temporary file location: {export_path}")

# Copy to Triton destination
os.makedirs(os.path.dirname(EXPORT_DESTINATION), exist_ok=True)
shutil.copy(export_path, EXPORT_DESTINATION)

print(f"Model copied to Triton repository: {EXPORT_DESTINATION}")
print(f"File size: {os.path.getsize(EXPORT_DESTINATION) / (1024*1024):.1f} MB")
