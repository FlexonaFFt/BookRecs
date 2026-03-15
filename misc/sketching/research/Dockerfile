FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY research/requirements.txt /app/research/requirements.txt
RUN pip install --no-cache-dir -r /app/research/requirements.txt

COPY . /app

CMD ["python", "-c", "print('Use docker compose run <service> ...')"]
