FROM python:3.12-slim

WORKDIR /app

COPY requirements-railway.txt .
RUN pip install --no-cache-dir -r requirements-railway.txt

COPY . .

ENV PYTHONPATH=/app

CMD python scripts/start_dashboard.py --host 0.0.0.0 --port ${PORT:-8000}
