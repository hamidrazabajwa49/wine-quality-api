FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas==3.0.2 \
    scikit-learn==1.8.0 \
    joblib==1.5.3 \
    fastapi==0.142.2 \
    uvicorn==0.54.0 \
    pydantic-settings==2.15.0 \
    streamlit==1.64.0 \
    requests==2.33.1

COPY src ./src
COPY data ./data

# Train inside the image so the pickled model matches the installed scikit-learn
RUN python -m wine_api.train

RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8000 8501

CMD ["uvicorn", "wine_api.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
