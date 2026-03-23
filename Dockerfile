FROM python:3.10-slim
ARG RUN_ID
RUN echo "Model Run ID: ${RUN_ID}"
WORKDIR /app
COPY train.py .
RUN pip install mlflow scikit-learn
CMD ["echo", "ready"]