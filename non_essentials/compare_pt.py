# #python compare_pt.py /home/testing/ocr_mob/bestan.pt /home/testing/ocr_mob/bestsh.pt --data /home/testing/ocr_mob/plate_detect/yolo_train/data.yaml



# import os
# import argparse
# import torch
# import time
# from tabulate import tabulate
# from ultralytics import YOLO
# from colorama import Fore, Style, init

# init(autoreset=True)

# def get_model_info(model_path):
#     """Extract metadata from YOLO model"""
#     try:
#         model = YOLO(model_path)
#         size_MB = os.path.getsize(model_path) / (1024 * 1024)
#         parameters = sum(p.numel() for p in model.model.parameters())
#         layers = len(list(model.model.modules()))
#         return {
#             "path": model_path,
#             "size_MB": round(size_MB, 2),
#             "parameters": parameters,
#             "layers": layers,
#         }
#     except Exception as e:
#         print(f"{Fore.RED}Error loading model {model_path}: {e}")
#         return None

# def evaluate_model(model_path, data_path):
#     """Run YOLO validation and measure speed"""
#     try:
#         model = YOLO(model_path)
#         start_time = time.time()
#         metrics = model.val(data=data_path, imgsz=640, verbose=False)
#         eval_time = time.time() - start_time

#         # Latency on dummy input
#         dummy_input = torch.zeros((1, 3, 640, 640))
#         t0 = time.time()
#         model(dummy_input)
#         latency = (time.time() - t0)

#         results = {
#             "mAP50": float(metrics.results_dict.get("metrics/mAP50(B)", 0)),
#             "mAP50-95": float(metrics.results_dict.get("metrics/mAP50-95(B)", 0)),
#             "precision": float(metrics.results_dict.get("metrics/precision(B)", 0)),
#             "recall": float(metrics.results_dict.get("metrics/recall(B)", 0)),
#             "eval_time_s": round(eval_time, 2),
#             "latency_s": round(latency, 3),
#         }
#         return results
#     except Exception as e:
#         print(f"{Fore.RED}Error evaluating model {model_path}: {e}")
#         return None

# def evaluate_model(model_path, data_
# def explain_best_model(best, other):
#     """Generate a textual explanation why one model is better"""
#     reasons = []
#     if best["mAP50-95"] > other["mAP50-95"]:
#         reasons.append(f"Higher mAP50-95 ({best['mAP50-95']} vs {other['mAP50-95']}) → better detection across thresholds")
#     if best["precision"] > other["precision"]:
#         reasons.append(f"Higher precision ({best['precision']} vs {other['precision']}) → fewer false positives")
#     if best["recall"] > other["recall"]:
#         reasons.append(f"Higher recall ({best['recall']} vs {other['recall']}) → detects more true objects")
#     if best["latency_s"] < other["lcision ({best['precision']} vs {other['precision']}) → fewer false positives")
#     if best["recall"] > other["recaatency_s"]:
#         reasons.append(f"Faster inference ({best['latency_s']}s vs {other['latency_s']}s per image)")
#     if best["parameters"] < other["parameters"]:
#         reasons.append(f"Smaller model size ({best['parameters']:,} vs {other['parameters']:,} parameters) → more efficient")

#     if not reasons:
#         reasons.append("Metrics are similar; no clear winner based on key metrics.")

#     return " | ".join(reasons)

# def compare_models(model_paths, data_path):
#     results = []
#     for path in model_paths:
#         print(f"\n🔍 Evaluating: {Fore.CYAN}{os.path.basename(path)}{Style.RESET_ALL}")
#         info = get_model_info(path)
#         if not info:
#             continue
#         metrics = evaluate_model(path, data_path)
#         if metrics:
#             info.update(metrics)
#         results.append(info)

#     if not results:
#         print(f"{Fore.RED}No models evaluated successfully.")
#         return

#     print("\n📊 Model Comparison Summary:\n")
#     print(tabulate(results, headers="keys", tablefmt="github", floatfmt=".4f"))

#     print("\n" + "=" * 70)cision ({best['precision']} vs {other['precision']}) → fewer false positives")
#     if best["recall"] > other["reca
#     print(f"{'MODEL PERFORMANCE SUMMARY':^70}")
#     print("=" * 70)

#     for r in results:
#         print(f"🧩 {os.path.basename(r['path'])}")
#         print(f"   🟩 mAP50-95: {r['mAP50-95']}")
#         print(f"   🟦 Precision: {r['precision']}")
#         print(f"   🟨 Recall: {r['recall']}")
#         print(f"   ⚡ Inference Latency: {r['latency_s']}s/image")
#         print(f"   🧠 Parameters: {r['parameters']:,}")
#         print(f"   💾 Size: {r['size_MB']} MB")
#         print(f"   ⏱️  Validation Time: {r['eval_time_s']}s")
#         print("-" * 70)

#     best = max(results, key=lambda x: x["mAP50-95"])
#     other = [r for r in results if r != best][0]

#     print(f"\n🏆 {Fore.GREEN}Best Model: {os.path.basename(best['path'])}{Style.RESET_ALL}")
#     explanation = explain_best_model(best, other)
#     print(f"   → Why better: {explanation}")
#     print("=" * 70 + "\n")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Compare two YOLOv8 .pt models on the same dataset")
#     parser.add_argument("model1", help="Path to first YOLO .pt file")
#     parser.add_argument("model2", help="Path to second YOLO .pt file")
#     parser.add_argument("--data", required=True, help="Path to dataset YAML file")
# cision ({best['precision']} vs {other['precision']}) → fewer false positives")
#     if best["recall"] > other["reca
#     args = parser.parse_args()
#     compare_models([args.model1, args.model2], args.data)


# #!/usr/bin/env python3
# """
# compare_yolov11_models.py

# Compare two YOLOv11 models (.pt) and produce a readable summary.
# Features:
# - Loads YOLOv11 checkpoints safely
# - Extracts parameters, layers, class names
# - Evaluates validation metrics (mAP50, mAP50-95, precision, recall)
# - Measures inference latency
# - Determines best model and explains why
# """

# import os
# import argparse
# import time
# from ultralytics import YOLO

# # ------------------------------
# # Utility: Get model info
# # ------------------------------
# def load_model_info(model_path):
#     """Load YOLOv11 model safely and extract metadata"""
#     model = YOLO(model_path)  # Safely loads YOLOv11 checkpoint
#     info = {
#         "path": model_path,
#         "size_MB": round(os.path.getsize(model_path) / (1024*1024), 2),
#         "parameters": sum(p.numel() for p in model.model.parameters()),
#         "layers": len(list(model.model.modules())),
#         "nc": getattr(model.model, "nc", "N/A"),
#         "names": getattr(model.model, "names", "N/A"),
#         "epochs": getattr(model, "epochs", "N/A"),
#     }
#     return model, info

# # ------------------------------
# # Utility: Evaluate model
# # ------------------------------
# def evaluate_model(model, data_yaml):
#     """Run validation and get mAP, precision, recall"""
#     results = model.val(data=data_yaml, split='val', verbose=False)
#     return {
#         "mAP50": results.box.map50,
#         "mAP50-95": results.box.map,
#         "precision": results.box.mp,
#         "recall": results.box.mr,
#         "eval_time_s": round(sum(results.speed.values()), 2)  # total time of preprocess+inference+postprocess

#     }

# # ------------------------------
# # Utility: Benchmark inference
# # ------------------------------
# def benchmark_inference(model):
#     """Run dummy inference to measure latency"""
#     import torch
#     dummy_img = torch.zeros((1, 3, 640, 640))
#     start = time.time()
#     _ = model.predict(dummy_img, imgsz=640, verbose=False)
#     latency = round(time.time() - start, 4)
#     return latency

# # ------------------------------
# # Main Comparison
# # ------------------------------
# def compare_models(model_paths, data_yaml):
#     summary = []
#     print("\n🔍 Evaluating models...\n")
    
#     for mp in model_paths:
#         print(f"📦 Loading model: {os.path.basename(mp)}")
#         model, info = load_model_info(mp)
#         metrics = evaluate_model(model, data_yaml)
#         latency = benchmark_inference(model)
        
#         combined = {**info, **metrics, "latency_s": latency}
#         summary.append(combined)

#     # ------------------------------
#     # Display readable summary
#     # ------------------------------
#     print("\n" + "="*70)
#     print("                     MODEL PERFORMANCE SUMMARY")
#     print("="*70)
#     for s in summary:
#         print(f"🧩 {os.path.basename(s['path'])}")
#         print(f"   🟩 mAP50-95       : {s['mAP50-95']:.4f}")
#         print(f"   🟦 Precision     : {s['precision']:.4f}")
#         print(f"   🟨 Recall        : {s['recall']:.4f}")
#         print(f"   ⚡ Inference Latency : {s['latency_s']}s/image")
#         print(f"   🧠 Parameters    : {s['parameters']}")
#         print(f"   💾 Size          : {s['size_MB']} MB")
#         print(f"   ⏱️  Validation Time : {s.get('val_time_s', 'N/A')}s")
#         print(f"   📊 Classes       : {len(s.get('names', []))}")
#         print("-"*70)

#     # ------------------------------
#     # Decide best model
#     # ------------------------------
#     # Priority: higher mAP50-95 → higher precision → lower latency
#     best_model = max(summary, key=lambda x: (x["mAP50-95"], x["precision"], -x["latency_s"]))
#     others = [s for s in summary if s["path"] != best_model["path"]]

#     print("\n🏆 Best Model: ", os.path.basename(best_model["path"]))
#     reason = f"Higher mAP50-95 ({best_model['mAP50-95']:.4f})"
#     for o in others:
#         if best_model["latency_s"] < o["latency_s"]:
#             reason += f" + Faster inference ({best_model['latency_s']}s vs {o['latency_s']}s)"
#     print("   → Why better:", reason)
#     print("="*70 + "\n")

# # ------------------------------
# # Command-line interface
# # ------------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Compare two YOLOv11 models")
#     parser.add_argument("model_a", help="Path to first YOLOv11 .pt model")
#     parser.add_argument("model_b", help="Path to second YOLOv11 .pt model")
#     parser.add_argument("--data", required=True, help="Path to data.yaml for validation")
#     args = parser.parse_args()

#     if not os.path.exists(args.model_a) or not os.path.exists(args.model_b):
#         print("❌ Model file(s) not found!")
#         return
#     if not os.path.exists(args.data):
#         print("❌ data.yaml not found!")
#         return

#     compare_models([args.model_a, args.model_b], args.data)

# if __name__ == "__main__":
#     main()

import os
import time
from ultralytics import YOLO
import torch

# ====================================
# CONFIG — just paste your paths here
# ====================================
MODEL_PATHS = [
    "/home/testing/mobile_ocr/train/runs/m300_16v11m_v1_exp/weights/best.pt",
    "/home/testing/mobile_ocr/train/runs/m50_16v11m_v2_finetune_exp/weights/best.pt",
    "/home/testing/mobile_ocr/bestsh.pt",
    "/home/testing/mobile_ocr/train/runs/m50_16v11m_v1_exp/weights/best.pt",
    "/home/testing/mobile_ocr/train/runs/m100_16v11m_v1_exp/weights/best.pt"
    # add more as needed...
]

DATA_YAML = "/home/testing/mobile_ocr/dataset/s3_196_dataset/data.yaml"  # path to your data.yaml


# ====================================
# Utilities
# ====================================

def load_model_info(model_path):
    """Load YOLOv11 model and extract metadata."""
    model = YOLO(model_path)
    info = {
        "path": model_path,
        "size_MB": round(os.path.getsize(model_path) / (1024*1024), 2),
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "nc": getattr(model.model, "nc", "N/A"),
        "names": getattr(model.model, "names", "N/A"),
    }
    return model, info


def evaluate_model(model, data_yaml):
    """Validate model and extract metrics."""
    results = model.val(data=data_yaml, split="val", verbose=False)
    return {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "precision": results.box.mp,
        "recall": results.box.mr,
        "eval_time_s": round(sum(results.speed.values()), 2),
    }


def benchmark_inference(model):
    """Benchmark model inference latency."""
    dummy_img = torch.zeros((1, 3, 640, 640))
    start = time.time()
    _ = model.predict(dummy_img, imgsz=640, verbose=False)
    latency = round(time.time() - start, 4)
    return latency


# ====================================
# Main comparison
# ====================================
def compare_models():
    summary = []

    print("\n🔍 Evaluating YOLO models...\n")
    for mp in MODEL_PATHS:
        if not os.path.exists(mp):
            print(f"❌ Model not found: {mp}")
            continue

        print(f"📦 Loading model: {os.path.basename(mp)}")
        model, info = load_model_info(mp)
        metrics = evaluate_model(model, DATA_YAML)
        latency = benchmark_inference(model)

        summary.append({**info, **metrics, "latency_s": latency})

    # Print results
    print("\n" + "=" * 70)
    print("                 MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    for s in summary:
        print(f"🧩 {os.path.basename(s['path'])}")
        print(f"   🟩 mAP50-95      : {s['mAP50-95']:.4f}")
        print(f"   🟦 Precision     : {s['precision']:.4f}")
        print(f"   🟨 Recall        : {s['recall']:.4f}")
        print(f"   ⚡ Latency       : {s['latency_s']}s/image")
        print(f"   💾 Size          : {s['size_MB']} MB")
        print(f"   🧠 Parameters    : {s['parameters']}")
        print(f"   📊 Classes       : {len(s.get('names', []))}")
        print("-" * 70)

    # Pick best
    best_model = max(summary, key=lambda x: (x["mAP50-95"], x["precision"], -x["latency_s"]))
    print("\n🏆 BEST MODEL:", os.path.basename(best_model["path"]))
    print(f"   → mAP50-95: {best_model['mAP50-95']:.4f}, Precision: {best_model['precision']:.4f}")
    print("=" * 70 + "\n")


# ====================================
# Run
# ====================================
if __name__ == "__main__":
    compare_models()
