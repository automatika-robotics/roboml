import time
import base64
import logging
import inspect

import pytest
import cv2
from roboml import models
from roboml.models._base import ModelTemplate
from roboml.interfaces import (
    VLLMInput,
    DetectionInput,
    SpeechToTextInput,
    TextToSpeechInput,
)


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
        model = Model(name="test")
        logging.info(f"Testing {Model.__name__}")
        model._initialize()
        for input in inputs:
            start_time = time.time()
            result = model._inference(data=input)
            if log_output:
                logging.info(result)
            logging.info("--- %s seconds ---" % (time.time() - start_time))


@pytest.mark.module("mllm")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_vllms(loaded_img, models_in_module):
    """
    Test vllms
    """
    inputs = []
    data = {
        "query": "What do you see?",
        "images": [loaded_img],
        "chat_history": True,
    }
    inputs.append(VLLMInput(**data))
    inputs.append(
        VLLMInput(query="How is it made?", images=[loaded_img], chat_history=True)
    )
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
    with open("tests/resources/test.wav", "rb") as file:
        file_bytes = file.read()
    data = {"query": file_bytes, "max_new_tokens": 100}
    inputs = [SpeechToTextInput(**data)]
    run_models(models_in_module, inputs)


@pytest.mark.module("vision")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_vison(loaded_img, models_in_module):
    """
    Test vision
    """
    data = {"images": [loaded_img], "threshold": 0.5}
    inputs = [DetectionInput(**data)]
    run_models(models_in_module, inputs, log_output=True)
