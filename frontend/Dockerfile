FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY frontend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir \
    -r requirements.txt

COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
