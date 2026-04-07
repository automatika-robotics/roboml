FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

# setup environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /roboml

COPY . .

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --break-system-packages .

# clean up source
RUN rm -rf /roboml
