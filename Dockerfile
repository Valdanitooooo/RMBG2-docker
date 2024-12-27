FROM python:3.10.11-slim-buster
MAINTAINER valdanito@qq.com

WORKDIR /app

RUN apt-get update
RUN apt-get install -y build-essential curl pkg-config git --fix-missing
RUN apt-get install -y libpoppler-dev poppler-utils poppler-data --fix-missing
RUN apt-get install -y libopencv-dev --fix-missing
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app

RUN pip install --no-cache-dir -r /app/requirements.txt
