FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train a default model at build time so API can serve immediately.
RUN python -m src.train --data-path data/spam.csv --model-version v1

EXPOSE 10000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "10000"]
