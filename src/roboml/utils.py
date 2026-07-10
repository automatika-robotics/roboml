import base64
import inspect
import logging
import os
from enum import Enum
from functools import wraps
from io import BytesIO
from typing import Callable, Optional, Union

import numpy as np
import torch
from PIL import Image as PILImage
from scipy.io.wavfile import write

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


class CheckpointSource(Enum):
    """Hub from which model checkpoints are downloaded."""

    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"


# First-party ModelScope alternatives for default checkpoints
# that are only available on HuggingFace Hub
MODELSCOPE_ALTERNATIVES = {
    "suno/bark-small": "microsoft/speecht5_tts",
    "suno/bark": "microsoft/speecht5_tts",
    "PekingU/rtdetr_r50vd_coco_o365": "facebook/detr-resnet-50",
}


def get_checkpoint_source(source: Optional[str] = None) -> str:
    """Get the effective checkpoint source.

    Precedence: explicit source param > ROBOML_SOURCE env var > huggingface.

    :param source:
    :type source: Optional[str]
    :rtype: str
    """
    source = (
        source or os.environ.get("ROBOML_SOURCE") or CheckpointSource.HUGGINGFACE.value
    )
    valid_sources = {s.value for s in CheckpointSource}
    if source not in valid_sources:
        raise ValueError(
            f"Invalid checkpoint source '{source}'. "
            f"Valid values are {sorted(valid_sources)}. "
            "Check the `source` init parameter or the ROBOML_SOURCE environment variable."
        )
    return source


def has_huggingface_credentials() -> bool:
    """Check if HuggingFace credentials are available, either through the
    HF_TOKEN environment variable or a cached `huggingface-cli login`.

    Returns True when credentials cannot be determined, leaving the decision
    to the download attempt.

    :rtype: bool
    """
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return True


def has_modelscope_credentials() -> bool:
    """Check if ModelScope credentials are available, either through the
    MODELSCOPE_API_TOKEN environment variable or a cached `modelscope login`.

    Returns True when credentials cannot be determined (e.g. older modelscope
    versions), leaving the decision to the download attempt.

    :rtype: bool
    """
    if os.environ.get("MODELSCOPE_API_TOKEN"):
        return True
    try:
        from modelscope_hub.config import HubConfig

        return bool(HubConfig().token)
    except Exception:
        return True


def resolve_checkpoint(
    checkpoint: str,
    source: Optional[str] = None,
    logger: logging.Logger = logger,
) -> str:
    """Resolve a checkpoint ID to something from_pretrained can load.

    For huggingface the checkpoint ID is passed through unchanged and
    downloading is left to the underlying library. For modelscope the
    checkpoint is downloaded with modelscope.snapshot_download and the
    local directory is returned.

    :param checkpoint:
    :type checkpoint: str
    :param source:
    :type source: Optional[str]
    :param logger:
    :type logger: logging.Logger
    :rtype: str
    """
    if get_checkpoint_source(source) != CheckpointSource.MODELSCOPE.value:
        return checkpoint

    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise ImportError(
            "Downloading checkpoints from ModelScope requires the modelscope package. "
            "Install it with: pip install roboml[modelscope]"
        ) from e

    logger.info(f"Downloading checkpoint {checkpoint} from ModelScope hub")
    try:
        checkpoint_dir = snapshot_download(checkpoint)
    except Exception as e:
        hint = MODELSCOPE_ALTERNATIVES.get(checkpoint)
        hint_msg = (
            f" '{checkpoint}' is not available on ModelScope. "
            f"Consider using '{hint}' instead, or set source to 'huggingface'."
            if hint
            else (
                " If the model is restricted on ModelScope, log in by setting "
                "the MODELSCOPE_API_TOKEN environment variable."
            )
        )
        raise RuntimeError(
            f"Failed to download checkpoint {checkpoint} from ModelScope.{hint_msg}"
        ) from e
    logger.info(f"Checkpoint downloaded to {checkpoint_dir}")
    return checkpoint_dir


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
