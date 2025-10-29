# # === Training Configuration ===

# # Experiment Configuration
# EXPERIMENT_NAME = "YOLOv11m_Training"

# # Model Configuration
# BASE_MODEL_PATH = ""

# # Dataset Configuration
# YAML_PATH = "/home/testing/mobile_ocr/dataset/yolo_train/data.yaml"

# # Training Parameters
# EPOCHS = 2
# BATCH_SIZE = 16
# IMG_SIZE = 640

# # Model Registration Name
# # Leave empty ("") for auto-naming with date/time format (e.g., DETECT_20251007_143052)
# # Provide a name (e.g., "plate_detector") for custom naming with auto-versioning (_v1, _v2, etc.)
# REGISTERED_MODEL_NAME = "test2"

# # MLflow Configuration
# MLFLOW_UI_PORT = 5004


# === Training Configuration ===

# Experiment Configuration
EXPERIMENT_NAME = "YOLOv11m_Training"

# Model Configuration
# Leave empty ("") to automatically download and use YOLOv11m base model
# Or provide path to a custom model (e.g., "/path/to/your/model.pt")
BASE_MODEL_PATH = ""

# Dataset Configuration
YAML_PATH = "/home/testing/mobile_ocr/dataset/lpds_v11yolov11/data.yaml"

# Training Parameters
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640

# Model Registration Name
# Leave empty ("") for auto-naming with date/time format (e.g., DETECT_20251007_143052)
# Provide a name (e.g., "plate_detector") for custom naming with auto-versioning (_v1, _v2, etc.)
REGISTERED_MODEL_NAME = "m100_16v11m"
# MLflow Configuration
MLFLOW_UI_PORT = 5004

# === Auto Model Download Logic ===
def get_model_path():
    """
    Returns the model path. If BASE_MODEL_PATH is empty,
    downloads YOLOv11m automatically. 
    """
    if BASE_MODEL_PATH and BASE_MODEL_PATH.strip():
        return BASE_MODEL_PATH
    else:
        print("📥 BASE_MODEL_PATH not provided. Using default YOLOv11m model...")
        return "yolo11m.pt"  # YOLO will auto-download this

# Use this in your training script
MODEL_TO_USE = get_model_path()