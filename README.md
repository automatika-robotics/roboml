# RoboML 🤖

RoboML is an aggregator package written for quickly deploying open source ML models for robots. It is designed to cover two basic use cases.

- **Readily deploy various useful models:** The package provides a wrapper around the 🤗 [**Transformers**](https://github.com/huggingface/transformers) and [**SentenceTransformers**](https://www.sbert.net/) libraries. Pretty much all relevant open source models from these libraries can be quickly deployed behind a highly scalable server endpoint.
- **Deploy Detection Models with Tracking**: With RoboML one can deploy all detection models available in [**MMDetection**](https://github.com/open-mmlab/mmdetection). An open source vision model aggregation library. These detection models can also be seemlesly used for tracking.
- **Use Open Source Vector DBs**: RoboML provides a unified interface for deploying Vector DBs along with ML models. Currently it is packaged with [**ChromaDB**](https://www.trychroma.com/) an open source multimodal vector database.
- **Aggregate robot specific ML models from the Robotics community**: RoboML aims to be an aggregator package of models trained by the robotics community. These models can range from Multimodal LLMs, vision models, or robot action models, and can be used with ROS based functional components. See the usage in [ROS Agents](https://automatika-robotics.github.io/ros-agents)

## Installation

RoboML has been tested on Ubuntu 20.04 and later. It should ideally be installed on a system with a GPU and CUDA 12.1. However, it should work without a GPU. If you encounter any installation problems, please open an issue.

`pip install roboml`

### From Source

```shell
git clone https://github.com/automatika-robotics/roboml.git && cd roboml
virtualenv venv && source venv/bin/activate
pip install pip-tools
pip install .
```

### For vision models support

If you want to utilize detection and tracking using Vision models from the MMDetection library, you will need to install a couple of dependancies as follows:

- Install roboml using the vision flag:

  `pip install roboml[vision]`

- Install mmcv using the installation instructions provided [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). For installation with pip, simply pick PyTorch and CUDA version that you have installed and copy the pip installation command generated. For example for PyTorch 2.1:

   `pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html`

- Install mmdetection as follows:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```

- Install ffmpeg and libGL are missing then run the following:

`sudo apt-get update && apt-get install ffmpeg libsm6 libxext6`

### Model quantization support

Roboml uses [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) for model quantization. However it is only installed as a dependency automatically on **x86_64** architectures as bitsandbytes pre-built wheels are not available for other architectures. For other architures, such as _aarch64_ on NVIDIA Jetson boards, it is recommended to build bitsandbytes from source using the following instructions:

```shell
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```
More details are available on the bitsandbytes [installation page](https://huggingface.co/docs/bitsandbytes/main/en/installation).

## Build in a Docker container (Recommended)

- Install docker desktop.
- Install [NVIDIA toolkit for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```shell
git clone https://github.com/automatika-robotics/roboml.git && cd roboml
# build the container image
docker build --tag=automatika:roboml .
# for NVIDIA Jetson boards replace the above command with
docker build --tag=automatika:roboml -f Dockerfile.Jetson .
# run the container with gpu support
docker run --runtime=nvidia --gpus all --rm -p 8000:8000 automatika:roboml
```

## Servers

By default roboml starts models as [ray serve](https://docs.ray.io/en/latest/serve/index.html) apps. Making the models scalable accross multiple infrastructure configurations. See [ray serve](https://docs.ray.io/en/latest/serve/index.html) for details.

### An Experimental Server based on RESP

When using ML models on robots, latency is a major consideration. When models are deployed on distributed infrastructure (and not on the edge, due to compute limitations), latency depends on both the model inference time and server communication time. Therefore, RoboML also implements an experimental server built using [RESP](https://github.com/antirez/RESP3) which can be accessed using any redis client. RESP is a human readable binary safe protocol, which is very simple to parse and thus can be used to implement servers significatly faster than HTTP, specially when the payloads are also packaged binary data (for example images, audio or video data). The RESP server uses msgpack a cross-platform library available in over 50 languages, to package data instead of JSON. Work on the server was inspired by earlier work of [@hansonkd](https://github.com/hansonkd) and his [Tino](https://github.com/hansonkd/Tino) project.

## Usage

To run an HTTP server simply run the following in the terminal

`roboml`

To run an RESP based server, run

`roboml-resp`

In order to see how these servers are called from a ROS package that implements its clients, please refer to [ROS Agents](https://automatika-robotics.github.io/ros-agents) documentation.

## Running Tests

To run tests, install with:

`pip install ".[dev]"`

And run the following in the root directory

`python -m pytest`

## Copyright

The code in this distribution is Copyright (c) 2024 Automatika Robotics unless explicitly indicated otherwise.

ROS Agents is made available under the MIT license. Details can be found in the [LICENSE](LICENSE) file.

## Contributions

ROS Agents has been developed in collaboration betweeen [Automatika Robotics](https://automatikarobotics.com/) and [Inria](https://inria.fr/). Contributions from the community are most welcome.
