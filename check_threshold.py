<<<<<<< HEAD
import sys
import mlflow

run_file = sys.argv[1]
threshold = float(sys.argv[2])

# Read Run ID
with open(run_file, "r") as f:
    run_id = f.read().strip()

client = mlflow.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0)

print(f"Run ID: {run_id}, Accuracy: {accuracy}")

if accuracy < threshold:
    print(f"Accuracy {accuracy} below threshold {threshold}. Failing...")
    sys.exit(1)
else:
    print("Threshold passed!")
=======
import mlflow
import os
import sys

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt") as f:
    run_id = f.read().strip()

run = mlflow.get_run(run_id)
acc = run.data.metrics["accuracy"]

print(f"Run ID: {run_id}")
print(f"Accuracy: {acc}")

if acc < 0.50:
    print("FAILED: Accuracy below threshold (0.50)")
    sys.exit(1)

print("PASSED: Accuracy meets threshold")
>>>>>>> a0fddbc5d91607e7a79a223388e8d90987b4ff9a
