
# Python 3.14 runtime
FROM python:3.14-slim

# Application directory
WORKDIR /app

# Python environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install inference dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application
COPY backend/ ./backend/

# Copy the saved model and preprocessor
COPY artifacts/ ./artifacts/

# Copy custom Python modules
COPY src/ ./src/
ENV PYTHONPATH=/app/src

# Expose the API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
