FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/cache/huggingface

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

RUN pip install --no-cache-dir \
    -r requirements.txt

COPY app ./app
COPY pdf_samples ./pdf_samples

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--app-dir", "app", "--host", "0.0.0.0", "--port", "8000"]
