FROM python:3.10.6-buster

WORKDIR /signlens

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY signlens signlens
COPY setup.py setup.py

COPY utils utils

COPY models_api models_api

RUN pip install .

CMD uvicorn signlens.api.fast:app --host 0.0.0.0 --port 8000
