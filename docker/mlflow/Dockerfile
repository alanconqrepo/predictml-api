FROM python:3.11-slim

RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    mlflow psycopg2-binary boto3

EXPOSE 5000
