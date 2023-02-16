FROM python:3.9

RUN apt-get update && apt-get install -y python3-opencv git

RUN python -m pip install \
  streamlit\ 
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
  seaborn  \
  opencv-python 

  # RUN git clone https://github.com/camlab-bioml/ccramic.git && cd ccramic && pip install .

COPY . app/ 

RUN cd app/ && pip install .

# ENTRYPOINT [ "ccramic" ]

