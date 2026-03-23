FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

RUN pip install --no-cache-dir mlflow scikit-learn

RUN echo "Downloading model for Run ID: ${RUN_ID}"

COPY train.py .

CMD ["python", "-c", "print('Model container is ready.')"]