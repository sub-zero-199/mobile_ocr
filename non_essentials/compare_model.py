import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("/home/testing/mobile_ocr/train/mlruns")
client = MlflowClient()

models = client.search_registered_models()
# filter_string="name LIKE 'm300%'"
all_model_versions = []

for model in models:
    print(f"\n{'='*60}")
    print(f"Model: {model.name}")
    print(f"{'='*60}")
    
    all_versions = client.search_model_versions(f"name='{model.name}'")
    
    if not all_versions:
        print("  No versions found for this model")
        continue
    
    for version in all_versions:
        print(f"\n  Version: {version.version}")
        print(f"  Run ID: {version.run_id}")
        print(f"  Stage: {version.current_stage}")
        
        try:
            run = client.get_run(version.run_id)
            print(f"  Metrics:")
            
            if run.data.metrics:
                metrics_data = {}
                for key, value in sorted(run.data.metrics.items()):
                    print(f"    {key}: {value:.4f}")
                    metrics_data[key] = value
                
                # Store for comparison
                all_model_versions.append({
                    'model_name': model.name,
                    'version': version.version,
                    'run_id': version.run_id,
                    'metrics': metrics_data
                })
            else:
                print("    No metrics logged in this run")
                
        except Exception as e:
            print(f"    Error: {e}")

print(f"\n{'='*60}")

# Find best model based on mAP50-95
if all_model_versions:
    print(f"\n{'='*60}")
    print("🏆 BEST MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    # Filter models that have mAP50-95 metric
    models_with_map = [m for m in all_model_versions if 'mAP50-95' in m['metrics']]
    
    if models_with_map:
        best_model = max(models_with_map, key=lambda x: x['metrics']['mAP50-95'])
        
        print(f"Best Model: {best_model['model_name']} (Version {best_model['version']})")
        print(f"Run ID: {best_model['run_id']}")
        print(f"\nKey Metrics:")
        print(f"  mAP50-95: {best_model['metrics'].get('mAP50-95', 0):.4f} ⭐")
        print(f"  mAP50: {best_model['metrics'].get('mAP50', 0):.4f}")
        print(f"  Precision: {best_model['metrics'].get('precision', 0):.4f}")
        print(f"  Recall: {best_model['metrics'].get('recall', 0):.4f}")
        
        print(f"\n{'='*60}")
        print("📊 All Models Ranked by mAP50-95:")
        print(f"{'='*60}")
        
        sorted_models = sorted(models_with_map, key=lambda x: x['metrics']['mAP50-95'], reverse=True)
        for idx, m in enumerate(sorted_models, 1):
            print(f"{idx}. {m['model_name']} v{m['version']}: mAP50-95={m['metrics']['mAP50-95']:.4f}")
    else:
        print("⚠️  No models found with mAP50-95 metric")
else:
    print("⚠️  No model versions with metrics found")

print(f"\n{'='*60}")