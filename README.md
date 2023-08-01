# Model Server

FastAPI and hypercorn based server for serving multimodal models.

`git clone https://gitlab.com/automatika/model_server.git`

`cd model_server`

Currently under development so:

`git checkout dev`

## Running with Docker (Recommended)

Install docker desktop.

Install [NVIDIA toolkit for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for the server)

### Build Image

`docker build --build-arg distro=<DISTRO> --tag=automatika:dist-ros-server --target=server .`

`docker run --runtime=nvidia --gpus all --rm -i -t automatika:dist-ros-server`

## Running Locally

`pip3 install -r req.txt`

### For Server



## Project status
Under Development
