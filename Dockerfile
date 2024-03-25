FROM python:3.10.6-buster

WORKDIR /signlens

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY signlens signlens
COPY utils utils
COPY setup.py setup.py

RUN pip install .

COPY processed_data/glossary.csv processed_data/glossary.csv
COPY models_api models_api

CMD uvicorn signlens.api.fast:app --host 0.0.0.0 --port 8000
