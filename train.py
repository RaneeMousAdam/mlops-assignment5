import mlflow
import random
# random.seed(42)

# # لو عندك secret هيتاخد من GitHub
# mlflow.set_tracking_uri("mlruns")

# with mlflow.start_run() as run:
#     accuracy = random.uniform(0.7, 0.95)

#     print(f"Accuracy: {accuracy}")
#     mlflow.log_metric("accuracy", accuracy)

#     # save run id
#     with open("model_info.txt", "w") as f:
#         f.write(run.info.run_id)

mlflow.set_tracking_uri("mlruns")
with mlflow.start_run() as run:
    accuracy = 0.88
    print(f"Accuracy: {accuracy}")
    mlflow.log_metric("accuracy", accuracy)

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)