FROM python:3.10-slim

ARG RUN_ID
<<<<<<< HEAD
RUN echo "Simulating download of model for Run ID: $RUN_ID"

CMD ["echo", "Model ready!"]
=======

RUN echo "Downloading model for Run ID: ${RUN_ID}"

WORKDIR /app

COPY train.py .

RUN pip install mlflow scikit-learn

CMD ["echo", "Model ready"]
>>>>>>> a0fddbc5d91607e7a79a223388e8d90987b4ff9a
