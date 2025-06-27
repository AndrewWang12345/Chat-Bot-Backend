# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model assets
COPY . .

# Expose port
EXPOSE 8181

# Run FastAPI app with Uvicorn
CMD ["python", "main.py"]
