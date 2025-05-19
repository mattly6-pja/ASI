FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV KEDRO_ENV=base

EXPOSE 8501

CMD kedro run && streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
