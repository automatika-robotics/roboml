<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/RoboML_DARK.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/_static/RoboML_LIGHT.png">
  <img alt="RoboML Logo" src="docs/_static/RoboML_LIGHT.png"  width="40%">
</picture>
<br/>

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

RoboML is an aggregator package for quickly deploying open-source ML models for robots. It supports three main use cases:

- **Rapid deployment of general-purpose models:** Wraps around popular ML libraries like 🤗 [**Transformers**](https://github.com/huggingface/transformers), allowing fast deployment of models through scalable server endpoints.
- **Deploy detection models with tracking:** Supports deployment of detection models from 🤗 [**Transformers**](https://huggingface.co/models?pipeline_tag=object-detection) (RT-DETR, DETR, Grounding DINO, etc.) with optional tracking integration.
- **Aggregate robot-specific models from the robotics community:** Intended as a platform for community-contributed multimodal models, usable in planning and control, especially with ROS components. See [EmbodiedAgents](https://automatika-robotics.github.io/embodied-agents).

## Models And Wrappers

| **Model Class**    | **Description**                                                                                                                           | **Default Checkpoint / Resource**                                                                                                                         | **Key Init Parameters**                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `TransformersLLM`  | General-purpose large language model (LLM) from [🤗 Transformers](https://github.com/huggingface/transformers)                            | [`Qwen/Qwen3-0.6B`](https://huggingface.co/models?other=LLM)                                                                             | `name`, `checkpoint`, `quantization`, `init_timeout`                                                  |
| `TransformersMLLM` | Multimodal vision-language model (MLLM) from [🤗 Transformers](https://github.com/huggingface/transformers)                               | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/models?pipeline_tag=image-text-to-text)                                                              | `name`, `checkpoint`, `quantization`, `init_timeout`                                                  |
| `RoboBrain2`       | Embodied planning + multimodal reasoning via [RoboBrain 2.0](https://github.com/FlagOpen/RoboBrain2.0)                                    | [`BAAI/RoboBrain2.0-3B`](https://huggingface.co/collections/BAAI/robobrain20-6841eeb1df55c207a4ea0036)                                                    | `name`, `checkpoint`, `init_timeout`                                                                  |
| `Whisper`          | Multilingual speech-to-text (ASR) from [OpenAI Whisper](https://openai.com/index/whisper)                                                 | `small.en` ([checkpoint list](https://github.com/SYSTRAN/faster-whisper/blob/d3bfd0a305eb9d97c08047c82149c1998cc90fcb/faster_whisper/transcribe.py#L606)) | `name`, `checkpoint`, `compute_type`, `init_timeout`                                                  |
| `TransformersTTS`  | Text-to-speech via [🤗 Transformers](https://huggingface.co/models?pipeline_tag=text-to-speech) (Bark, VITS, SpeechT5, SeamlessM4T, etc.) | [`suno/bark-small`](https://huggingface.co/models?pipeline_tag=text-to-speech)                                                                            | `name`, `checkpoint`, `voice`, `vocoder_checkpoint`, `init_timeout`                                   |
| `VisionModel`      | Detection + tracking via [🤗 Transformers](https://huggingface.co/models?pipeline_tag=object-detection)                                   | [`PekingU/rtdetr_r50vd_coco_o365`](https://huggingface.co/models?pipeline_tag=object-detection)                                                         | `name`, `checkpoint`, `setup_trackers`, `tracking_distance_threshold`, `num_trackers`, `init_timeout` |

## Installation

RoboML has been tested on Ubuntu 20.04 and later. A GPU with CUDA 12.1+ is recommended. If you encounter problems, please [open an issue](https://github.com/automatika-robotics/roboml/issues).

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

VisionModel uses HuggingFace Transformers for object detection and tracking. It works out of the box with any [HuggingFace object detection model](https://huggingface.co/models?pipeline_tag=object-detection) (RT-DETR, DETR, Grounding DINO, YOLOS, etc.). Object tracking via [ByteTrack](https://github.com/roboflow/trackers) is included.

If `ffmpeg` or `libGL` is missing:

```bash
sudo apt-get update && apt-get install ffmpeg libsm6 libxext6
```

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
