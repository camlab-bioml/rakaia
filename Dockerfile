FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils \
    python3-pip python3-opencv \
    git libmysqlclient-dev pkg-config libperl-dev libgtk-3-dev libnotify-dev \
    libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0 libvips \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install .

ENV DISPLAY=:0.0
ENV DLClight=True

EXPOSE 5000
