import os
from ultralytics import YOLO
import config
from pathlib import Path
import cv2

# =========================
# CONFIGURATIONt
# =========================
MODEL_PATH = "/home/testing/mobile_ocr/bestsh.pt"
INPUT_PATH = "/home/testing/mobile_ocr/test_img"  # single image or folder
OUTPUT_DIR = "/home/testing/mobile_ocr/inference_results"
IMG_SIZE = config.IMG_SIZE  # 640
CONF_THRESHOLD = 0.80
IOU_THRESHOLD = 0.45

# create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images_with_boxes")
LABEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(LABEL_OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
print(f"📥 Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# =========================
# INFERENCE FUNCTION
# =========================
def run_inference(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.*"))
    else:
        print(f"❌ Invalid input path: {input_path}")
        return

    for file in files:
        print(f"\n🔎 Processing: {file.name}")
        results = model.predict(
            source=str(file),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            save=False,  # we handle saving manually
            verbose=False
        )

        # Save labels
        txt_file = os.path.join(LABEL_OUTPUT_DIR, file.stem + ".txt")
        with open(txt_file, "w") as f:
            for r in results:
                for box in r.boxes.data.tolist():  # xyxy, conf, cls
                    x1, y1, x2, y2, conf, cls = box
                    f.write(f"{int(cls)} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n")

        # Draw boxes and save image
        annotated_img = results[0].plot()  # returns numpy array with boxes
        img_save_path = os.path.join(IMG_OUTPUT_DIR, file.name)
        cv2.imwrite(img_save_path, annotated_img)
        print(f"✅ Saved annotated image + labels for: {file.name}")

# =========================
# RUN INFERENCE
# =========================
if __name__ == "__main__":
    run_inference(INPUT_PATH)
    print(f"\n🎯 All predictions saved to: {OUTPUT_DIR}")
