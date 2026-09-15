FROM python:3.14.2-slim
WORKDIR /app
RUN pip install flask gunicorn mlflow scikit-learn
COPY predict.py .
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]