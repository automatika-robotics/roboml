FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as model_server

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /roboml

COPY . .

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6 && pip install pip-tools && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install .[vision]

RUN pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

WORKDIR /

RUN git clone https://github.com/open-mmlab/mmdetection.git && pip install -v -e mmdetection/

WORKDIR /roboml

ENTRYPOINT ["roboml"]
