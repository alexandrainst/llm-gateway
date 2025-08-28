FROM python:3.12-slim
WORKDIR /app

# Install uv for fast venv creation
RUN pip install uv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONUNBUFFERED=1
# Command is specified in docker-compose.yml for flexibility
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
