# 1. Use official Python base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy all files to the container
COPY . /app

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose FastAPI port
EXPOSE 8200

# 6. Command to run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8200"]

