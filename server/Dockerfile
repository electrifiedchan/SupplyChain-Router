FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# HF Docker Spaces require port 7860
EXPOSE 7860

# Start the environment server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
