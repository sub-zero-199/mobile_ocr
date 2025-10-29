import os
import torch
import mlflow
import mlflow.pytorch
from ultralytics import YOLO
from mlflow.tracking import MlflowClient
from datetime import datetime
import config


def get_model_path():
    """Return the path to the pretrained model for finetuning."""
    if config.BASE_MODEL_PATH and config.BASE_MODEL_PATH.strip():
        print(f"📂 Using pretrained model: {config.BASE_MODEL_PATH}")
        return config.BASE_MODEL_PATH
    else:
        print("📥 BASE_MODEL_PATH not provided. Using default YOLOv11m model...")
        return "yolo11m.pt"


def get_model_name(base_name, client):
    """Generate versioned model name for registration."""
    base_name = base_name.strip()
    if not base_name:
        date_str = datetime.now().strftime("%Y%m%d")
        search_prefix = date_str
    else:
        search_prefix = f"{base_name}_v"

    try:
        models = client.search_registered_models(filter_string=f"name LIKE '{search_prefix}%'")
        existing_versions = []
        for model in models:
            model_name = model.name
            if model_name.startswith(search_prefix):
                suffix = model_name.replace(search_prefix, "")
                try:
                    version_num = int(suffix)
                    existing_versions.append(version_num)
                except ValueError:
                    continue

        next_version = 1
        if existing_versions:
            next_version = max(existing_versions) + 1

        if not base_name:
            return f"{date_str}_v{next_version}" if models else date_str
        else:
            return f"{base_name}_v{next_version}"

    except Exception as e:
        print(f"⚠️ Error checking existing models: {e}")
        if not base_name:
            return datetime.now().strftime("%Y%m%d")
        else:
            return f"{base_name}_v1"


def finetune_model():
    """Main finetuning function."""
    print("\n" + "="*60)
    print("🤖 YOLOv11m FINETUNING PIPELINE")
    print("="*60)

    # === Get model path ===
    model_path = get_model_path()

    # === MLflow setup ===
    train_dir = os.path.join(os.getcwd(), "train")
    mlflow_dir = os.path.join(train_dir, "mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    client = MlflowClient()

    # Generate model name
    registered_model_name = get_model_name(config.REGISTERED_MODEL_NAME, client)
    print(f"\n📦 Model will be registered as: {registered_model_name}")

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # === Start finetuning with MLflow logging ===
    with mlflow.start_run(run_name=f"{registered_model_name}_Finetune") as run:
        print(f"\n🚀 Starting finetuning for {registered_model_name}...")

        mlflow.log_params({
            "pretrained_model": model_path,
            "registered_name": registered_model_name,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "img_size": config.IMG_SIZE,
            "dataset": os.path.basename(config.YAML_PATH)
        })

        # Load pretrained YOLOv11 model
        print(f"\n🔄 Loading model: {model_path}")
        model = YOLO(model_path)

        # Finetune
        results = model.train(
            data=config.YAML_PATH,
            epochs=config.EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.BATCH_SIZE,
            device="auto",
            project=os.path.join(train_dir, "runs"),
            name=f"{registered_model_name}_finetune_exp",
            plots=True,
            exist_ok=True,
            augment=True,    # enable augmentation for small dataset
            lr0=0.001,       # low learning rate for finetuning
            pretrained=True  # start from pretrained weights
        )

        # Log metrics
        metrics_dict = results.results_dict if hasattr(results, 'results_dict') else {}
        final_metrics = {
            "mAP50": metrics_dict.get('metrics/mAP50(B)', 0),
            "mAP50-95": metrics_dict.get('metrics/mAP50-95(B)', 0),
            "precision": metrics_dict.get('metrics/precision(B)', 0),
            "recall": metrics_dict.get('metrics/recall(B)', 0),
        }
        mlflow.log_metrics(final_metrics)
        print(f"📊 Metrics: {final_metrics}")

        # Log & register model
        best_model_path = os.path.join(train_dir, "runs", f"{registered_model_name}_finetune_exp", "weights", "best.pt")
        if os.path.exists(best_model_path):
            logged_model = mlflow.pytorch.log_model(model.model, artifact_path="model")

            # Log plots
            for pf in ['confusion_matrix.png', 'results.png', 'F1_curve.png']:
                plot_path = os.path.join(train_dir, "runs", f"{registered_model_name}_finetune_exp", pf)
                if os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, artifact_path="plots")

            # Register model
            registered_model = mlflow.register_model(
                model_uri=logged_model.model_uri,
                name=registered_model_name,
                tags={
                    "task": "object_detection",
                    "framework": "YOLOv11",
                    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            )
            print(f"✅ Registered: {registered_model.name} v{registered_model.version}")

    return mlflow_dir, client


def select_best_model(mlflow_dir, client):
    """Select best model for production based on mAP50-95."""
    print(f"\n{'='*60}")
    print("🏆 SELECTING BEST MODEL FOR PRODUCTION")
    print(f"{'='*60}")

    models = client.search_registered_models()
    all_versions = []

    for model in models:
        for version in client.search_model_versions(f"name='{model.name}'"):
            try:
                run = client.get_run(version.run_id)
                if run.data.metrics:
                    map_value = run.data.metrics.get('mAP50-95')
                    if map_value is not None and map_value > 0:
                        all_versions.append({
                            'model_name': model.name,
                            'version': version.version,
                            'mAP50-95': map_value
                        })
            except Exception as e:
                print(f"⚠️ Error reading model {model.name} v{version.version}: {e}")

    if all_versions:
        best = max(all_versions, key=lambda x: x['mAP50-95'])

        # Remove previous production aliases
        for model in models:
            for version in client.search_model_versions(f"name='{model.name}'"):
                if "production" in version.aliases:
                    client.delete_registered_model_alias(model.name, "production")

        # Set new production model
        client.set_registered_model_alias(best['model_name'], "production", best['version'])
        print(f"🚀 PRODUCTION: {best['model_name']} v{best['version']} (mAP50-95: {best['mAP50-95']:.4f})")
    else:
        print("⚠️ No models with mAP50-95 found")

    print(f"\n🌐 View MLflow UI: mlflow ui --backend-store-uri file:{mlflow_dir} --port {config.MLFLOW_UI_PORT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    mlflow_dir, client = finetune_model()
    select_best_model(mlflow_dir, client)
