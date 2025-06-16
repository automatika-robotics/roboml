import base64
import inspect
import logging
from enum import Enum
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from importlib.metadata import distribution, PackageNotFoundError
from scipy.io.wavfile import write

from roboml.tools.download import DownloadManager

logger = logging.getLogger("roboml")


def pre_process_images_to_pil(
    data: Union[list[str], list[np.ndarray]],
    concatenate: bool = False,
) -> Union[PILImage.Image, list[PILImage.Image]]:
    """
    Returns PIL Image given an np array or base64 str
    :param data: list of images as np.ndarray or base64 str
    :type data: list[np.ndarray] | list[str]
    :param concatenate: bool
    :rtype: PILImage.Image | list[PILImage.Image]
    """
    # TODO: Handle multiple images by concatenation
    if concatenate:
        if isinstance(data[0], np.ndarray):
            return PILImage.fromarray(data[0])
        return PILImage.open(BytesIO(base64.b64decode(data[0])))
    if isinstance(data[0], np.ndarray):
        return [PILImage.fromarray(img) for img in data]
    return [PILImage.open(BytesIO(base64.b64decode(img))) for img in data]


def pre_process_images_to_np(
    data: Union[list[str], list[np.ndarray]],
    concatenate: bool = False,
) -> Union[np.ndarray, list[np.ndarray]]:
    """
    Returns numpy array given an np array or base64 str
    :param data: list of images as np.ndarray or base64 str
    :type data: list[np.ndarray] | list[str]
    :param concatenate: bool
    :rtype: np.ndarray | list[np.ndarray]
    """
    # TODO: Handle multiple images by concatenation
    if concatenate:
        if isinstance(data[0], np.ndarray):
            return data[0]
        return np.array(PILImage.open(BytesIO(base64.b64decode(data[0]))))
    if isinstance(data[0], np.ndarray):
        # assume the whole list is ndarray
        return data  # type: ignore
    return [np.array(PILImage.open(BytesIO(base64.b64decode(img)))) for img in data]


def b64_str_to_bytes(data: str) -> bytes:
    """
    Returns bytes given a str
    :param data: base64 encoded str
    :type data: str
    :rtype: bytes
    """
    return base64.b64decode(data)


def post_process_audio(
    data: torch.Tensor | np.ndarray, sample_rate: int = 16000, get_bytes: bool = False
) -> Union[str, bytes]:
    """
    Returns a bye file location given a torch tensor of audio
    :param      data:  torch tensor
    :type       data:  torch.Tensor
    :returns:   file location
    :rtype:     str
    """
    # create numpy array
    if not isinstance(data, np.ndarray):
        data = data.detach().numpy().squeeze().astype(np.float32)

    # open buffer and write to it with hard coded sampling rate
    bytes_wav = bytes()
    byte_io = BytesIO(bytes_wav)
    write(byte_io, sample_rate, data)
    audio_bytes = byte_io.read()

    if get_bytes:
        return audio_bytes

    return base64.b64encode(audio_bytes).decode("utf-8")


class Quantization(Enum):
    """Model Quantization."""

    EIGHT = "8bit"
    FOUR = "4bit"


def get_quantization_config(level: Optional[str], logger: logging.Logger = logger):
    """Utility method to create BitsAndBytesConfig for model quantization.

    :param level:
    :type level: Optional[str]
    :param logger:
    :type logger: logging.Logger
    :rtype: Optional[BitsAndBytesConfig]
    """
    from transformers import BitsAndBytesConfig

    # If cuda not available, skip quantization
    if not torch.cuda.is_available():
        logger.warning("Cuda not detected, quantization settings will not be applied.")
        return None

    if level == Quantization.FOUR.value:
        logger.info("Loading model with 4bit quantization")
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    elif level == Quantization.EIGHT.value:
        logger.info("Loading model with 8bit quantization")
        return BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16
        )
    else:
        logger.info("Loading unquantized model")
        return None


class Status(Enum):
    """Status for model nodes."""

    LOADED = 1
    INITIALIZING = 2
    READY = 3
    INITIALIZATION_ERROR = 4


def background_task(function: Callable):
    """Generic decorator to mark functions that should be run as background tasks.
    :param function:
    :type function: Callable
    """

    @wraps(function)
    def _wrapper(*a, **kw):
        """_wrapper.
        :param a:
        :param kw:
        """
        return function(*a, **kw)

    return _wrapper


def is_background_task(func: Callable) -> bool:
    """Helper method to check if a callable is decorated as a background task.
    :param func:
    :type func: Callable
    :rtype: bool
    """
    decorators = [
        i.strip()
        for i in inspect.getsource(func).split("\n")
        if i.strip().startswith("@")
    ]
    return "@background_task" in decorators


def get_mmdet_model(
    cache: str | Path, checkpoint: str, logger: logging.Logger = logger
) -> tuple[str, str]:
    """Helper method written to avoid the use of MIM for downloading checkpoints.
    Checks with the checkpoint exists in cache, if not downloads it.
    :param cache:
    :type cache: str | os.PathLike
    :param checkpoint:
    :type checkpoint: str
    :param logger:
    :type logger: logging.Logger
    :rtype: tuple[str, str]
    """
    # create cache dir if it does not exist
    if isinstance(cache, str):
        cache = Path(cache)
    cache.mkdir(exist_ok=True)

    # check if checkpoint exists
    checkpoint_dir = cache / Path(checkpoint)
    if checkpoint_dir.is_dir():
        config = checkpoint_dir / Path(f"{checkpoint}.py")
        weights = checkpoint_dir / Path(f"{checkpoint}.pth")
        if config.is_file() and weights.is_file():
            logger.info(f"{checkpoint} found in cache.")
            return str(config), str(weights)

    # get mmdet package path and build dict of all models available
    pkg_path = _get_mmdet_package_info()
    all_models = _build_mmdet_model_dict(pkg_path)

    try:
        model_info = all_models[checkpoint]
    except KeyError:
        logger.error(
            f"Model metadata not found in mmdetection package. Make sure you have the correct version installed which contains {checkpoint}"
        )
        raise

    config_from_pkg = pkg_path / Path(model_info["Config"])
    weights_url = model_info["Weights"]

    # create checkpoint dir if it does not exist
    checkpoint_dir.mkdir(exist_ok=True)

    # config cache destination
    config = checkpoint_dir / Path(f"{checkpoint}.py")

    # read config file to get its base files and dump in cache folder
    cfg = _load_mm_config(config_from_pkg)

    # TODO: Add config modifications here
    _dump_mm_config(cfg, config)
    logger.info(f"Config file copied to {config}")

    # download weights
    weights = checkpoint_dir / Path(f"{checkpoint}.pth")
    downloader = DownloadManager(weights_url, weights)
    downloader.download()

    return str(config), str(weights)


def convert_with_mmdeploy(
    model_config: str,
    weights: str,
    device: str,
    cache: str | Path,
    config_type="detection_tensorrt_dynamic-320x320-1344x1344",
):
    """Convert with mmdeploy"""
    try:
        from mmdeploy.utils import get_input_shape
        from mmdeploy.apis.utils import build_task_processor
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "To use tensorrt deployment, make sure that mmdeploy and mmdeploy-runtime-gpu are installed. They can be installed on x86_64 systems with `pip install mmdeploy==1.3.1 mmdeploy-runtime-gpy==1.3.1`. For other architectures refer to the following link for instructions on building from source: https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/01-how-to-build/build_from_source.md"
        ) from e

    mmdeploy_config, config_path = _get_mmdeploy_config(
        cache, model_config, config_type
    )

    backend_files = _generate_backend(cache, config_path, model_config, weights, device)

    model_cfg = _load_mm_config(model_config)

    task_processor = build_task_processor(model_cfg, mmdeploy_config, device)

    model = task_processor.build_backend_model(
        backend_files, task_processor.update_data_preprocessor
    )

    input_shape = get_input_shape(mmdeploy_config)

    return model, input_shape, task_processor


def _get_mmdet_package_info() -> Path:
    """_get_mmdet_package_info.

    :rtype: Path
    """
    # get location of mmdet package
    try:
        package = distribution("mmdet")
        location = Path(str(package.locate_file("")))
    except PackageNotFoundError:
        logger.error(
            "MMDetection does not seem to be installed. Please install it to run object detection models."
        )
        raise
    # return mim path for the mmdet package
    mim_path = location / Path("mmdet") / Path(".mim")
    if not mim_path.exists():
        logger.error(
            "MMDetection does not seem to be installed. Please install it to run object detection models."
        )
        raise PackageNotFoundError
    return mim_path


def _build_mmdet_model_dict(pkg_path: Path) -> dict:
    """_build_mmdet_model_dict.

    :param pkg_path:
    :type pkg_path: Path
    :rtype: dict
    """
    all_models = {}
    try:
        with open(pkg_path / Path("model-index.yml"), "rb") as f:
            available_models = yaml.load(f, Loader=yaml.SafeLoader)

        for model_meta_path in available_models["Import"]:
            with open(pkg_path / Path(model_meta_path), "rb") as f:
                model_metainfo = yaml.load(f, Loader=yaml.SafeLoader)
            if model_list := model_metainfo.get("Models"):
                for m in model_list:
                    m_config = m.get("Config")
                    m_weights = m.get("Weights")
                    if not (m_config and m_weights):
                        continue
                    all_models[m.get("Name")] = {
                        "Config": m_config,
                        "Weights": m_weights,
                    }
    except Exception as e:
        logger.error(
            "Exception occurred with reading models metadata from mmdetection package, make sure mmdetection was correctly installed"
        )
        raise e
    return all_models


def _load_mm_config(config: Path | str):
    """_load_mm_config.

    :param config:
    :type config: Path
    :rtype: Config
    """
    from mmengine import Config

    try:
        cfg = Config.fromfile(Path(config))
    except Exception as e:
        logger.error(
            f"Exception occurred when read config. Make sure mmengine is correctly installed. {e}"
        )
        raise
    return cfg


def _dump_mm_config(cfg, config_path: Path) -> None:
    """_dump_mm_config.

    :param cfg:
    :type cfg: Config
    :param config_path:
    :type config_path: Path
    :rtype: None
    """
    try:
        cfg.dump(config_path)
    except Exception as e:
        logger.error(
            f"Exception occurred when dumping config. Make sure mmengine is correctly installed. {e}"
        )
        raise


def _get_mmdeploy_config(cache: str | Path, model_config: str, config_type: str):
    model_dir = Path(model_config).parent
    config_path = model_dir / Path(f"{config_type}.py")
    # check if mmdeploy config already exists in model dir
    if model_dir.is_dir():
        if config_path.is_file():
            logger.info(f"MMdeploy config {config_type} found in cache.")
            return _load_mm_config(config_path), config_path

    # check mmdeploy directory in cache dir
    cache = Path(cache)
    mmdeploy_dir = cache / Path("mmdeploy-1.3.1")
    config = mmdeploy_dir / Path(f"configs/mmdet/detection/{config_type}.py")

    # check if mmdeploy config already exists in mmdeploy_dir
    if mmdeploy_dir.is_dir():
        if config.is_file():
            logger.info(f"MMdeploy config {config_type} found in cache.")
            _config = _load_mm_config(config)
            _dump_mm_config(_config, config_path)
            return _config, config_path

    # Download mmdeploy zip
    logger.info(
        f"MMdeploy config {config_type} not found in cache. Attempting to download..."
    )
    zip_file = cache / Path("mmdeploy.zip")

    if not zip_file.is_file():
        downloader = DownloadManager(
            "https://github.com/open-mmlab/mmdeploy/archive/refs/tags/v1.3.1.zip",
            zip_file,
        )
        downloader.download()

    # unzip mmdeploy
    import zipfile

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        logger.info("Extracting downloaded mmdeploy ...")
        zip_ref.extractall(cache)
    if config.is_file():
        _config = _load_mm_config(config)
        _dump_mm_config(_config, config_path)
        return _config, config_path
    else:
        raise Exception("Could not load mmdeploy config.")


def _generate_backend(
    cache: str | Path,
    mmdeploy_config: str | Path,
    model_config: str,
    weights: str,
    device: str,
) -> list[str]:
    """Generate tensorrt backend"""
    try:
        import torch

        logger.info("CUDA available", torch.cuda.is_available())
        logger.info("cuDNN enabled", torch.backends.cudnn.enabled)
        import tensorrt
        import pycuda

        logger.info("tensorrt version", tensorrt.__version__)
        logger.info("pycuda version", pycuda.VERSION_TEXT)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "To use tensorrt deployment, tensorrt, pycuda and cudnn must be installed. You can follow the installation procedure described here: "
        ) from e

    deploy_dir = Path(model_config).parent / Path("tensorrt")
    engine_file = deploy_dir / Path("end2end.engine")
    if engine_file.is_file():
        logging.info(f"Found cached tensorrt engine file {engine_file}")
        return [str(engine_file)]

    logger.info("Converting pytorch model to tensorrt...")
    cache = Path(cache)
    deploy_script = cache / Path("mmdeploy-1.3.1") / Path("tools/deploy.py")

    import subprocess

    # Start conversion
    subprocess.run([
        "python",
        f"{deploy_script}",
        f"{mmdeploy_config}",
        f"{model_config}",
        f"{weights}",
        "tests/resources/test.jpeg",
        "--work-dir",
        f"{deploy_dir}",
        "--device",
        f"{device}",
    ])

    return [str(engine_file)]
