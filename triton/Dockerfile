FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for cache efficiency
COPY sources/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and static files
COPY sources/ /app/

# The ONNX model will be mounted via docker-compose or copied
# For now we assume it's copied or mounted to /app/model.onnx

EXPOSE 8080

CMD ["python", "triton.py"]
