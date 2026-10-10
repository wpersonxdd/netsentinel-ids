# Analyst console + CLI in one image.
#   docker build -t netsentinel .
#   docker run -p 8501:8501 netsentinel                     # trains on synthetic data at build time, serves the UI
#   docker run -v "$PWD/data:/app/data" netsentinel python -m nids all --data data/raw/official
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN pip install --no-cache-dir -e . --no-deps \
 && python -m nids all

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
