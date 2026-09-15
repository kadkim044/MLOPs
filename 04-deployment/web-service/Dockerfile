FROM python:3.14.2-slim

WORKDIR /app

RUN pip install flask gunicorn scikit-learn pandas

COPY 04-deployment/web-service/predict.py .
COPY models/lin_reg.bin .

CMD ["gunicorn", "-b", "0.0.0.0:9696", "predict:app"]