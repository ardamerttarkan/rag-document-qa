## Qdrant Başlatma

docker compose up -d

## FastAPI Çalıştırma

PYTHONPATH=app uvicorn api:app \
 --reload \
 --host 127.0.0.1 \
 --port 8000

## Streamlit Arayüzü Çalıştırma

streamlit run frontend/streamlit_app.py
