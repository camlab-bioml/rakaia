### 1. Get Linux
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && \
#     apt-get -y install default-jre-headless && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*
#

RUN apt-get update && apt-get install -y python3-pip python3-opencv \
git libmysqlclient-dev pkg-config libperl-dev libgtk-3-dev libnotify-dev \
libsdl2-mixer-2.0-0 libsdl2-image-2.0-0 libsdl2-2.0-0

RUN apt-get install -y python3.9

# RUN apt-get -y install build-essential libgtk-3-dev
#
# RUN apt update
# RUN apt -y upgrade
# RUN apt install -y make gcc build-essential libgtk-3-dev wget git
# RUN apt install -y openjdk-11-jdk-headless default-libmysqlclient-dev libnotify-dev libsdl2-dev libwebkit2gtk-4.0-dev
#
# ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# ENV VIRTUAL_ENV=/opt/venv
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# RUN pip install --upgrade pip
# RUN pip install wheel cython numpy
#
# RUN pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04 wxPython==4.1.0

# RUN python3 -m pip install attrdict cellprofiler

  # RUN git clone https://github.com/camlab-bioml/ccramic.git && cd ccramic && pip install .

COPY . app/

WORKDIR /app/

RUN pip install -U -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04 wxPython==4.2.1

RUN pip install -r requirements.txt && pip install .

ENV DISPLAY=:0.0

ENV DLClight=True

EXPOSE 5000

# RUN python3 -m pip install cellprofiler

# ENTRYPOINT [ "chmod", "+x", "bash", "/scripts/run_ccramic.sh" ]
