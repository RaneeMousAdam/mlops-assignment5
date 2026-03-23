FROM python:3.10-slim

ARG RUN_ID
RUN echo "Simulating download of model for Run ID: $RUN_ID"

CMD ["echo", "Model ready!"]