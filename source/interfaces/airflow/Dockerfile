FROM apache/airflow:2.10.5-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends docker.io && rm -rf /var/lib/apt/lists/*
USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-docker==3.11.0 \
    apache-airflow-providers-postgres==5.13.0 \
    psycopg2-binary==2.9.10
