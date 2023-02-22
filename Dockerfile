### 1. Get Linux
FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install default-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3.9 python3-pip python3-opencv git libmysqlclient-dev pkg-config

RUN python3 -m pip install \
  numpy \
  streamlit \ 
  pillow \
  pandas \
  matplotlib \
  fpdf \
  streamlit-aggrid \
  pytest \
  freeport \
  st-clickable_images \
  streamlit-drawable-canvas \
  numpy \
  scikit-image \
  anndata \
  scanpy \
  phenograph \
  seaborn

RUN apt-get -y install build-essential libgtk-3-dev

RUN apt update
RUN apt -y upgrade
RUN apt install -y make gcc build-essential libgtk-3-dev wget git
RUN apt install -y openjdk-11-jdk-headless default-libmysqlclient-dev libnotify-dev libsdl2-dev libwebkit2gtk-4.0-dev

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install wheel cython numpy

RUN pip install -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04 wxPython==4.1.0

# RUN python3 -m pip install attrdict cellprofiler 

  # RUN git clone https://github.com/camlab-bioml/ccramic.git && cd ccramic && pip install .

COPY . app/ 

RUN cd app/ && pip install .

RUN python3 -m pip install cellprofiler


# ENTRYPOINT [ "ccramic" ]

