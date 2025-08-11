FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y libosmesa6-dev patchelf

RUN pip install uv
