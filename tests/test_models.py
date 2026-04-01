"""GPU-based integration tests: test actual model loading and inference.
These require a GPU and will download model weights on first run.
"""

import time
import base64
import logging
import pytest
import cv2

from roboml.models import (
    TransformersLLM,
    TransformersMLLM,
    TransformersTTS,
    VisionModel,
    RoboBrain2,
)
from roboml.interfaces import (
    LLMInput,
    VLLMInput,
    PlanningInput,
    DetectionInput,
    TextToSpeechInput,
)


@pytest.fixture
def loaded_img():
    """Fixture to load test image as base64 string."""
    img = cv2.imread("tests/resources/test.jpeg", cv2.COLOR_BGR2RGB)
    encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    _, buffer = cv2.imencode(".png", img, encode_params)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def run_model(model_cls, init_kwargs=None, inputs=None, log_output=False):
    """Initialize a model and run inference on given inputs."""
    model = model_cls(logger=logging.getLogger("test"))
    logging.info(f"Testing {model_cls.__name__}")
    if init_kwargs:
        model._initialize(**init_kwargs)
    else:
        model._initialize()
    for data in inputs:
        start_time = time.time()
        result = model._inference(data=data)
        if log_output:
            logging.info(result)
        logging.info("--- %s seconds ---" % (time.time() - start_time))
    return result


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_llm():
    """Test TransformersLLM with default checkpoint."""
    data = LLMInput(query=[{"role": "user", "content": "Whats up?"}])
    result = run_model(TransformersLLM, inputs=[data], log_output=True)
    assert "output" in result
    assert isinstance(result["output"], str)
    assert len(result["output"]) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_mllm(loaded_img):
    """Test TransformersMLLM with default checkpoint."""
    data = VLLMInput(
        query=[{"role": "user", "content": "What do you see?"}],
        images=[loaded_img],
    )
    result = run_model(TransformersMLLM, inputs=[data], log_output=True)
    assert "output" in result
    assert isinstance(result["output"], str)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_vision(loaded_img):
    """Test VisionModel with default RT-DETRv2 checkpoint."""
    data = DetectionInput(images=[loaded_img], threshold=0.5)
    result = run_model(VisionModel, inputs=[data], log_output=True)
    assert "output" in result
    assert isinstance(result["output"], list)
    # Should detect at least something in the test image
    if result["output"]:
        det = result["output"][0]
        assert "bboxes" in det
        assert "labels" in det
        assert "scores" in det


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_vision_with_tracking(loaded_img):
    """Test VisionModel with object tracking enabled."""
    data = DetectionInput(
        images=[loaded_img],
        threshold=0.3,
        labels_to_track=["person"],
    )
    result = run_model(
        VisionModel,
        init_kwargs={"setup_trackers": True, "num_trackers": 1},
        inputs=[data],
        log_output=True,
    )
    assert "output" in result


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_tts_bark():
    """Test TransformersTTS with Bark (generative waveform model)."""
    data = TextToSpeechInput(query="This text should be spoken aloud.", get_bytes=False)
    result = run_model(
        TransformersTTS,
        init_kwargs={"checkpoint": "suno/bark-small"},
        inputs=[data],
    )
    assert "output" in result
    assert isinstance(result["output"], str)
    # Should be valid base64
    decoded = base64.b64decode(result["output"])
    assert len(decoded) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_tts_vits():
    """Test TransformersTTS with VITS (forward-only waveform model)."""
    data = TextToSpeechInput(query="Hello world.", get_bytes=False)
    result = run_model(
        TransformersTTS,
        init_kwargs={"checkpoint": "facebook/mms-tts-eng", "voice": None},
        inputs=[data],
    )
    assert "output" in result
    assert isinstance(result["output"], str)
    decoded = base64.b64decode(result["output"])
    assert len(decoded) > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_planning(loaded_img):
    """Test RoboBrain2 planning model. Requires HF_TOKEN for gated model access."""
    data_general = PlanningInput(
        query=[{"role": "user", "content": "What is in this image?"}],
        task="general",
        images=[loaded_img],
    )
    data_grounding = PlanningInput(
        query=[{"role": "user", "content": "The sandwich"}],
        task="grounding",
        images=[loaded_img],
    )
    result = run_model(
        RoboBrain2,
        inputs=[data_general, data_grounding],
        log_output=True,
    )
    assert "output" in result
    assert "thinking" in result
