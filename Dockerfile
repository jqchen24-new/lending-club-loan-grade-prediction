FROM python:3.10-slim

WORKDIR /app

COPY requirements-api.txt .

# Install PyTorch CPU-only version — much smaller (~800MB vs 2GB+)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (API-only)
RUN pip install --no-cache-dir -r requirements-api.txt

COPY predictor.pkl .
COPY model.py .
COPY predict.py .
COPY explain.py .
COPY background.csv .

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]