<picture>
  <source media="(prefers-color-scheme: dark)" srcset="_static/RoboML_DARK.png">
  <source media="(prefers-color-scheme: light)" srcset="_static/RoboML_LIGHT.png">
  <img alt="RoboML Logo" src="_static/RoboML_LIGHT.png"  width="40%">
</picture>
<br/><br/>

[![English Version][en-badge]][en-url]
[![PyPI][pypi-badge]][pypi-url]
[![MIT licensed][mit-badge]][mit-url]
[![Python Version][python-badge]][python-url]

[en-badge]: https://img.shields.io/badge/docs-English-blue.svg
[en-url]: ../README.md
[pypi-badge]: https://img.shields.io/pypi/v/roboml.svg
[pypi-url]: https://pypi.org/project/roboml/
[mit-badge]: https://img.shields.io/pypi/l/roboml.svg
[mit-url]: https://github.com/automatika-robotics/roboml/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/roboml.svg
[python-url]: https://www.python.org/downloads/

<br/>

RoboML 是一个聚合包，用于快速部署面向机器人的开源机器学习模型。它设计用于满足两个基本用例：

- **快速部署各种实用模型：** 本包封装了多个常用的机器学习库，如 🤗 [**Transformers**](https://github.com/huggingface/transformers)，可以快速将这些库中大多数开源模型部署在高度可扩展的服务端点上。
- **部署检测与跟踪模型：** 通过 RoboML，您可以部署来自 [**MMDetection**](https://github.com/open-mmlab/mmdetection) 的所有检测模型，并支持无缝集成跟踪功能。
- **聚合机器人社区的专用模型：** RoboML 旨在成为一个机器人社区训练模型的聚合平台，尤其关注多模态模型在 ROS 控制与规划中的应用。参见 [ROS Agents](https://automatika-robotics.github.io/ros-agents) 了解更多信息。

## 模型与包装器

| **モデルクラス名** | **説明**                                                                                                         | **デフォルトチェックポイント / リソース**                                                                                                                                                                 | **主な初期化パラメータ**                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TransformersLLM`  | 一般的な大規模言語モデル（LLM）。[🤗 Transformers](https://github.com/huggingface/transformers) に基づく         | [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/models?other=LLM)                                                                                                                             | `name`、`checkpoint`、`quantization`、`init_timeout`                                                                                                                 |
| `TransformersMLLM` | マルチモーダル画像・言語モデル（MLLM）。[🤗 Transformers](https://github.com/huggingface/transformers) に基づく  | [`HuggingFaceM4/idefics2-8b`](https://huggingface.co/models?pipeline_tag=image-text-to-text)                                                                                                              | `name`、`checkpoint`、`quantization`、`init_timeout`                                                                                                                 |
| `RoboBrain2`       | [RoboBrain 2.0](https://github.com/FlagOpen/RoboBrain2.0) による長期計画とマルチモーダル推論モデル               | [`BAAI/RoboBrain2.0-7B`](https://huggingface.co/collections/BAAI/robobrain20-6841eeb1df55c207a4ea0036)                                                                                                    | `name`、`checkpoint`、`init_timeout`                                                                                                                                 |
| `Whisper`          | [OpenAI Whisper](https://openai.com/index/whisper) による多言語音声認識（ASR）モデル                             | `small.en`（[チェックポイント一覧](https://github.com/SYSTRAN/faster-whisper/blob/d3bfd0a305eb9d97c08047c82149c1998cc90fcb/faster_whisper/transcribe.py#L606)）                                           | `name`、`checkpoint`、`compute_type`、`init_timeout`                                                                                                                 |
| `SpeechT5`         | [Microsoft SpeechT5](https://github.com/microsoft/SpeechT5) によるテキスト読み上げ（TTS）モデル                  | `microsoft/speecht5_tts`                                                                                                                                                                                  | `name`、`checkpoint`、`voice`、`init_timeout`                                                                                                                        |
| `Bark`             | [SunoAI Bark](https://github.com/suno-ai/bark) によるテキスト読み上げ（TTS）モデル                               | [`suno/bark-small`](https://huggingface.co/collections/suno/bark-6502bdd89a612aa33a111bae)、[音声の種類](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) | `name`、`checkpoint`、`voice`、`attn_implementation`、`init_timeout`                                                                                                 |
| `MeloTTS`          | [MeloTTS](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) による多言語音声合成モデル | 言語: `EN`、話者ID: `EN-US`                                                                                                                                                                               | `name`、`language`、`speaker_id`、`init_timeout`                                                                                                                     |
| `VisionModel`      | [MMDetection](https://github.com/open-mmlab/mmdetection) に基づく物体検出・追跡モデル                            | [`dino-4scale_r50_8xb2-12e_coco`](https://github.com/open-mmlab/mmdetection?tab=readme-ov-file#overview-of-benchmark-and-model-zoo)                                                                       | `name`、`checkpoint`、`setup_trackers`、`cache_dir`、`tracking_distance_function`、`tracking_distance_threshold`、`deploy_tensorrt`、`_num_trackers`、`init_timeout` |

## 安装

RoboML 已在 Ubuntu 20.04 及更高版本中测试过。推荐在配有 GPU 且支持 CUDA 12.1 或以上版本的系统上安装。如遇安装问题，请提交 issue。

```bash
pip install roboml
```

### 从源码安装

```bash
git clone https://github.com/automatika-robotics/roboml.git && cd roboml
virtualenv venv && source venv/bin/activate
pip install pip-tools
pip install .
```

## 支持视觉模型

如果您希望使用 MMDetection 库中的视觉检测与跟踪模型，请按以下步骤安装依赖项：

- 使用 vision 选项安装 RoboML：

  ```bash
  pip install roboml[vision]
  ```

- 按照 [mmcv 安装说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 进行安装。以 PyTorch 2.1 和 CUDA 12.1 为例：

  ```bash
  pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
  ```

- 安装 mmdetection：

  ```bash
  git clone https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  pip install -v -e .
  ```

- 如果系统缺少 ffmpeg 和 libGL，请执行以下命令：

  ```bash
  sudo apt-get update && apt-get install ffmpeg libsm6 libxext6
  ```

### 基于 TensorRT 的模型部署

RoboML 中的视觉模型可通过 NVIDIA TensorRT 进行加速推理（需具备 NVIDIA GPU 支持）。当前仅支持基于 Linux 的 x86_64 系统。请参考 [官方安装指南](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)。

## Docker 构建（推荐方式）

建议 NVIDIA Jetson 用户在 Docker 容器中运行 RoboML。

- 安装 Docker Desktop
- 安装 [NVIDIA Docker 工具包](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
git clone https://github.com/automatika-robotics/roboml.git && cd roboml

# 构建容器镜像
docker build --tag=automatika:roboml .
# 或者 Jetson 用户请替换命令为
docker build --tag=automatika:roboml -f Dockerfile.Jetson .

# 启动 HTTP 服务，支持 GPU
docker run --runtime=nvidia --gpus all --rm -p 8000:8000 automatika:roboml roboml
# 启动 RESP 服务
docker run --runtime=nvidia --gpus all --rm -p 6379:6379 automatika:roboml roboml-resp
```

- 可选项：可将主机缓存目录挂载到容器以缓存模型：

  ```bash
  -v ~/.cache:/root/.cache
  ```

## 服务端

RoboML 默认作为 [ray serve](https://docs.ray.io/en/latest/serve/index.html) 应用运行，可在不同基础设施配置间扩展模型能力。

### WebSocket 接口

支持 WebSocket 输入输出流，特别适合用于语音识别（STT）与语音合成（TTS）模型。

### 实验性 RESP 服务端

机器人使用模型时对延迟非常敏感。在模型部署在远程服务器时，通信时间与模型推理时间都至关重要。RoboML 实现了基于 [RESP](https://github.com/antirez/RESP3) 的实验性服务器，可通过 Redis 客户端访问。

RESP 是一种人类可读的二进制安全协议，解析简单，速度快。该服务使用跨平台 [msgpack](https://msgpack.org/) 而非 JSON 传输数据，可显著提升图像、音频、视频等二进制数据的传输效率。该服务器受 [@hansonkd](https://github.com/hansonkd) 的 [Tino 项目](https://github.com/hansonkd/Tino) 启发开发。

## 使用方式

运行 HTTP 服务：

```bash
roboml
```

运行 RESP 服务：

```bash
roboml-resp
```

欲了解如何在 ROS 包中调用这些服务，请参见 [ROS Agents 文档](https://automatika-robotics.github.io/ros-agents)。

## 运行测试

使用如下命令安装开发依赖：

```bash
pip install ".[dev]"
```

运行测试：

```bash
python -m pytest
```

## 版权声明

除非另有说明，本项目代码版权归 © 2024 Automatika Robotics 所有。
RoboML 基于 MIT 许可协议发布，详情见 [LICENSE](LICENSE) 文件。

## 社区贡献

ROS Agents 项目由 [Automatika Robotics](https://automatikarobotics.com/) 与 [Inria](https://inria.fr/) 合作开发，欢迎社区贡献！
