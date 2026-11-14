# One image for the boss and the workers; docker-compose.yml picks the command.
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mapreduce/ mapreduce/
RUN python -m mapreduce.protogen

ENV PYTHONUNBUFFERED=1 \
    MR_DATA=/data
