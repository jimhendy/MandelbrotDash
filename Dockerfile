FROM python:3.10.9-slim-bullseye as build-image

WORKDIR app

COPY requirements.txt .

RUN python -m pip install -U pip
RUN pip install -r requirements.txt

COPY ./ .

EXPOSE 3003
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:3003", "--log-level=debug", "application:server"]