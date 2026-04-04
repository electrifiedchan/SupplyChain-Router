FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# HF Docker Spaces require port 7860
EXPOSE 7860

# server.app:app → WORKDIR /app, so Python finds server/app.py correctly
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
