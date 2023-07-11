Bootstrap: docker
From: ubuntu:22.04
Stage: apps

%files
    ccramic/
    tests/
    setup.py
    README.md

%post
export DEBIAN_FRONTEND=noninteractive
# apt-get update && \
    # apt-get -y install default-jre-headless && \
    # apt-get clean && \
    # rm -rf /var/lib/apt/lists/*

apt-get update && apt-get install -y python3.9 python3-pip python3-opencv git libmysqlclient-dev pkg-config

# apt-get -y install build-essential libgtk-3-dev

# apt update
# apt -y upgrade
# apt install -y make gcc build-essential libgtk-3-dev wget git
# apt install -y openjdk-11-jdk-headless default-libmysqlclient-dev libnotify-dev libsdl2-dev libwebkit2gtk-4.0-dev

pip install --upgrade pip
pip install wheel cython numpy

# export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# export VIRTUAL_export=/opt/vexport
# export PATH="$VIRTUAL_export/bin:$PATH"

python3 -m pip install .
