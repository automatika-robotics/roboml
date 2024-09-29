import base64
import inspect
import logging
import logging.config
from enum import Enum
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from pkg_resources import get_distribution
from scipy.io.wavfile import write

from roboml.tools.download import DownloadManager

# Considering utils is in src/roboml, get path to log config
PROJECT_ROOT = Path(__file__).parent.parent.parent
logging.config.fileConfig(PROJECT_ROOT / Path("log.ini"))
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
        return [img for img in data if isinstance(img, np.ndarray)]
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
    data: torch.Tensor, sample_rate: int = 16000, get_bytes: bool = False
) -> Union[str, bytes]:
    """
    Returns a bye file location given a torch tensor of audio
    :param      data:  torch tensor
    :type       data:  torch.Tensor
    :returns:   file location
    :rtype:     str
    """
    # create numpy array
    np_data = data.cpu().detach().numpy().squeeze().astype(np.float32)

    # open buffer and write to it with hard coded sampling rate
    bytes_wav = bytes()
    byte_io = BytesIO(bytes_wav)
    write(byte_io, sample_rate, np_data)
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


def get_mmdet_model(cache: str | Path, checkpoint: str) -> tuple[str, str]:
    """Helper method written to avoid the use of MIM for downloading checkpoints.
    Checks with the checkpoint exists in cache, if not downloads it.
    :param cache:
    :type cache: str | os.PathLike
    :param checkpoint:
    :type checkpoint: str
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
            logging.info(f"{checkpoint} found in cache.")
            return str(config), str(weights)

    # get mmdet package path and build dict of all models available
    pkg_path = _get_mmdet_package_info()
    all_models = _build_mmdet_model_dict(pkg_path)

    try:
        model_info = all_models[checkpoint]
    except KeyError:
        logging.error(
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
    cfg = _load_mmdet_config(config_from_pkg)

    # TODO: Add config modifications here
    _dump_mmdet_config(cfg, config)
    logging.info(f"Config file copied to {config}")

    # download weights
    weights = checkpoint_dir / Path(f"{checkpoint}.pth")
    downloader = DownloadManager(weights_url, weights)
    downloader.download()

    return str(config), str(weights)


def _get_mmdet_package_info() -> Path:
    """_get_mmdet_package_info.

    :rtype: Path
    """
    # get location of mmdet package
    package = get_distribution("mmdet")
    if not package.location:
        logging.error(
            "MMDetection does not seem to be installed. Please install it to run object detection models."
        )
        raise Exception
    package_path = Path(package.location) / Path("mmdet")
    if not package_path.exists():
        logging.error(
            "MMDetection does not seem to be installed. Please install it to run object detection models."
        )
        raise Exception

    # return mim path for the mmdet package
    mim_path = package_path / Path(".mim")
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
        logging.error(
            "Exception occured with reading models metadata from mmdetection package, make sure mmdetection was correctly installed"
        )
        raise e
    return all_models


def _load_mmdet_config(config: Path):
    """_load_mmdet_config.

    :param config:
    :type config: Path
    :rtype: Config
    """
    from mmengine import Config

    try:
        cfg = Config.fromfile(config)
    except Exception as e:
        logging.error(
            f"Exception occured when read config. Make sure mmengine is correctly installed. {e}"
        )
        raise
    return cfg


def _dump_mmdet_config(cfg, config_path: Path) -> None:
    """_dump_mmdet_config.

    :param cfg:
    :type cfg: Config
    :param config_path:
    :type config_path: Path
    :rtype: None
    """
    try:
        cfg.dump(config_path)
    except Exception as e:
        logging.error(
            f"Exception occured when dumping config. Make sure mmengine is correctly installed. {e}"
        )
        raise
