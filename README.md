<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/RoboML_DARK.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/RoboML_LIGHT.png">
  <img alt="RoboML Logo" src="docs/_static/RoboML_LIGHT.png"  width="40%">
</picture>
<br/><br/>

[![中文版本][cn-badge]][cn-url]
[![PyPI][pypi-badge]][pypi-url]
[![MIT licensed][mit-badge]][mit-url]
[![Python Version][python-badge]][python-url]

[cn-badge]: https://img.shields.io/badge/文档-中文-blue.svg
[cn-url]: docs/README.zh-CN.md
[pypi-badge]: https://img.shields.io/pypi/v/roboml.svg
[pypi-url]: https://pypi.org/project/roboml/
[mit-badge]: https://img.shields.io/pypi/l/roboml.svg
[mit-url]: https://github.com/automatika-robotics/roboml/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/roboml.svg
[python-url]: https://www.python.org/downloads/
<br/>

RoboML is an aggregator package for quickly deploying open-source ML models for robots. It supports three main use cases:

- **Rapid deployment of general-purpose models:** Wraps around popular ML libraries like 🤗 [**Transformers**](https://github.com/huggingface/transformers), allowing fast deployment of models through scalable server endpoints.
- **Deploy detection models with tracking:** Supports deployment of all detection models in [**MMDetection**](https://github.com/open-mmlab/mmdetection) with optional tracking integration.
- **Aggregate robot-specific models from the robotics community:** Intended as a platform for community-contributed multimodal models, usable in planning and control, especially with ROS components. See [ROS Agents](https://automatika-robotics.github.io/ros-agents).

## Installation

RoboML has been tested on Ubuntu 20.04 and later. A GPU with CUDA 12.1+ is recommended. If you encounter issues, please [open an issue](https://github.com/automatika-robotics/roboml/issues).

```bash
pip install roboml
```

### From Source

```bash
git clone https://github.com/automatika-robotics/roboml.git && cd roboml
virtualenv venv && source venv/bin/activate
pip install pip-tools
pip install .
```

## Vision Model Support

To use detection and tracking features via MMDetection:

- Install RoboML with the vision extras:

  ```bash
  pip install roboml[vision]
  ```

- Install `mmcv` using the appropriate CUDA and PyTorch versions as described in [their docs](https://mmcv.readthedocs.io/en/latest/get_started/installation.html). Example for PyTorch 2.1 with CUDA 12.1:

  ```bash
  pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
  ```

- Install `mmdetection`:

  ```bash
  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  pip install -v -e .
  ```

- If `ffmpeg` or `libGL` is missing:

  ```bash
  sudo apt-get update && apt-get install ffmpeg libsm6 libxext6
  ```

### TensorRT-Based Model Deployment

RoboML vision models can optionally be accelerated with NVIDIA TensorRT on Linux x86_64 systems. For setup, follow the [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

## Docker Build (Recommended)

Jetson users are especially encouraged to use Docker.

- Install Docker Desktop
- Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
git clone https://github.com/automatika-robotics/roboml.git && cd roboml

# Build container image
docker build --tag=automatika:roboml .
# For Jetson boards:
docker build --tag=automatika:roboml -f Dockerfile.Jetson .

# Run HTTP server
docker run --runtime=nvidia --gpus all --rm -p 8000:8000 automatika:roboml roboml
# Or run RESP server
docker run --runtime=nvidia --gpus all --rm -p 6379:6379 automatika:roboml roboml-resp
```

- (Optional) Mount your cache dir to persist downloaded models:

  ```bash
  -v ~/.cache:/root/.cache
  ```

## Servers

RoboML uses [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) to host models as scalable apps across various environments.

### WebSocket Endpoint

WebSocket endpoints are exposed for streaming use cases (e.g., STT/TTS).

### Experimental RESP Server

For ultra-low latency in robotics, RoboML also includes a RESP-based server compatible with any Redis client.
RESP (see [spec](https://github.com/antirez/RESP3)) is a lightweight, binary-safe protocol. Combined with `msgpack` instead of JSON, it enables very fast I/O, ideal for binary data like images, audio, or video.

This work is inspired by [@hansonkd](https://github.com/hansonkd)’s [Tino project](https://github.com/hansonkd/Tino).

## Usage

Run the HTTP server:

```bash
roboml
```

Run the RESP server:

```bash
roboml-resp
```

Example usage in ROS clients is documented in [ROS Agents](https://automatika-robotics.github.io/ros-agents).

## Running Tests

Install dev dependencies:

```bash
pip install ".[dev]"
```

Run tests from the project root:

```bash
python -m pytest
```

## Copyright

Unless otherwise specified, all code is © 2024 Automatika Robotics.
RoboML is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contributions

ROS Agents is developed in collaboration between [Automatika Robotics](https://automatikarobotics.com/) and [Inria](https://inria.fr/). Community contributions are welcome!
