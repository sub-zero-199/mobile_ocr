import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("/home/testing/mobile_ocr/train/mlruns")
client = MlflowClient()

models = client.search_registered_models()
# filter_string="name LIKE 'm300%'"

for model in models:
    print(f"\n{'='*60}")
    print(f"Model: {model.name}")
    print(f"{'='*60}")
    
    # Get all versions for this model
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
                # Show all metrics
                for key, value in sorted(run.data.metrics.items()):
                    print(f"    {key}: {value:.4f}")
            else:
                print("    No metrics logged in this run")
                
        except Exception as e:
            print(f"    Error: {e}")

print(f"\n{'='*60}")