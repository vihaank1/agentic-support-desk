FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi "uvicorn[standard]" && \
    pip install --no-cache-dir openai google-genai
COPY src ./src
COPY data ./data
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
# Cloud Run injects $PORT; default 8080 for local runs
CMD ["sh", "-c", "uvicorn support_desk.service:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}"]
