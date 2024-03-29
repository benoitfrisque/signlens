FROM tensorflow/tensorflow

WORKDIR /signlens

COPY requirements_api.txt requirements.txt

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential libpq-dev

COPY signlens signlens
COPY setup.py setup.py

RUN pip install .

COPY processed_data/glossary.csv processed_data/glossary.csv
COPY processed_data/glossary_decoding.csv processed_data/glossary_decoding.csv
COPY models_api models_api

CMD uvicorn signlens.api.fast:app --host 0.0.0.0 --port $PORT
