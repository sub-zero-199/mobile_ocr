import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:///home/testing/ocr_mob/plate_detect/mlruns")
client = MlflowClient()

# List all runs in the experiment
experiment = client.get_experiment_by_name("YOLOv11m_Training")
if experiment:
    runs = client.search_runs(experiment.experiment_id)
    print(f"Found {len(runs)} runs")
    for run in runs[:3]:  # Show last 3 runs
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Metrics: {run.data.metrics}")