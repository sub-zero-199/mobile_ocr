from ultralytics import YOLO
import os

# ---------------- CONFIGURATION ----------------
pt_model_path = "/home/testing/mobile_ocr/final.pt"  # Path to your .pt model
output_dir = "/home/testing/mobile_ocr/custom_models"  # Destination folder
onnx_filename = "final_model.onnx"                  # Output ONNX filename

# Create destination folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ---------------- LOAD YOLO MODEL ----------------
model = YOLO(pt_model_path)

# ---------------- EXPORT TO ONNX ----------------
model.export(
    format="onnx",      # Export format
    dynamic=True,       # Allow dynamic batch/shape
    simplify=True,      # Simplify ONNX graph
    project=output_dir, # Folder to save the exported model
    name=onnx_filename  # Filename of the exported model
)

print(f"✅ Model exported successfully to: {os.path.join(output_dir, onnx_filename)}")
