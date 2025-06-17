import time
import base64
import logging
import inspect
import pytest
import cv2
import wave

from roboml import models
from roboml.models.vision import VisionModel
from roboml.models._base import ModelTemplate
from roboml.interfaces import (
    LLMInput,
    VLLMInput,
    DetectionInput,
    SpeechToTextInput,
    TextToSpeechInput,
)


def wav_to_base64(filepath):
    with wave.open(filepath, "rb") as wf:
        # Ensure format is 16-bit PCM and 16000Hz
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit audio (2 bytes per sample)")
        if wf.getframerate() != 16000:
            raise ValueError("Expected 16000 Hz sample rate")

        # Read raw frames
        audio_bytes = wf.readframes(wf.getnframes())

        # Encode to base64
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        return base64_audio


@pytest.fixture
def loaded_img():
    """Fixture to load test image"""
    img = cv2.imread("tests/resources/test.jpeg", cv2.COLOR_BGR2RGB)
    encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    _, buffer = cv2.imencode(".png", img, encode_params)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


@pytest.fixture
def models_in_module(request):
    """Get models in a module"""
    module = request.node.get_closest_marker("module")
    if not module:
        return None
    module = module.args[0]
    if module == "vision":
        return [VisionModel]
    module = getattr(models, module)
    return [
        model_class
        for _, model_class in inspect.getmembers(module, predicate=inspect.isclass)
        if issubclass(model_class, ModelTemplate) and model_class is not ModelTemplate
    ]


def run_models(models_in_module, inputs, log_output=False):
    """
    Init models and run inference
    """
    for Model in models_in_module:
        model = Model(logger=logging.getLogger("test"))
        logging.info(f"Testing {Model.__name__}")
        model._initialize()
        for input in inputs:
            start_time = time.time()
            result = model._inference(data=input)
            if log_output:
                logging.info(result)
            logging.info("--- %s seconds ---" % (time.time() - start_time))


@pytest.mark.module("llm")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_llms(models_in_module):
    """
    Test vllms
    """
    data = {
        "query": [{"role": "user", "content": "Whats up?"}],
    }
    input = LLMInput(**data)
    inputs = [input]
    run_models(models_in_module, inputs, log_output=True)


@pytest.mark.module("mllm")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_vllms(loaded_img, models_in_module):
    """
    Test vllms
    """
    data = {
        "query": [{"role": "user", "content": "What do you see?"}],
        "images": [loaded_img],
    }
    input = VLLMInput(**data)
    inputs = [input]
    run_models(models_in_module, inputs, log_output=True)


@pytest.mark.module("text_to_speech")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_text_to_speech(models_in_module):
    """
    Test text to speech
    """
    data = {"query": "This text should be spoken aloud.", "get_bytes": False}
    inputs = [TextToSpeechInput(**data)]
    run_models(models_in_module, inputs)


@pytest.mark.module("speech_to_text")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_speech_to_text(models_in_module):
    """
    Test speech to text
    """
    wav_str = wav_to_base64("tests/resources/test.wav")
    data = {"query": wav_str, "max_new_tokens": None}
    inputs = [SpeechToTextInput(**data)]
    run_models(models_in_module, inputs, log_output=True)


@pytest.mark.module("vision")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_vison(loaded_img, models_in_module):
    """
    Test vision
    """
    data = {"images": [loaded_img], "threshold": 0.5}
    inputs = [DetectionInput(**data)]
    run_models(models_in_module, inputs, log_output=True)
