"""GPU-based integration tests: test actual model loading and inference.
These require a GPU and will download model weights on first run.
"""

import time
import base64
import logging
import pytest
import cv2
import numpy as np

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
def test_llm_from_modelscope():
    """Test TransformersLLM pulling its checkpoint from ModelScope.
    Requires the modelscope package (pip install roboml[modelscope])."""
    pytest.importorskip("modelscope")
    data = LLMInput(query=[{"role": "user", "content": "Whats up?"}])
    result = run_model(
        TransformersLLM,
        init_kwargs={"source": "modelscope"},
        inputs=[data],
        log_output=True,
    )
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
def test_mllm_video():
    """Test TransformersMLLM with video input given as a frame array."""
    import numpy as np

    img = cv2.cvtColor(cv2.imread("tests/resources/test.jpeg"), cv2.COLOR_BGR2RGB)
    frames = np.stack([img] * 8)
    data = VLLMInput(
        query=[{"role": "user", "content": "What do you see in this video?"}],
        videos=[frames],
        video_fps=4.0,
        max_video_frames=8,
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
    """Test RoboBrain2 planning model with the default checkpoint."""
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
    data_pointing = PlanningInput(
        query=[{"role": "user", "content": "The pickles"}],
        task="pointing",
        images=[loaded_img],
    )
    data_affordance = PlanningInput(
        query=[{"role": "user", "content": "Pick up the sandwich"}],
        task="affordance",
        images=[loaded_img],
    )
    data_trajectory = PlanningInput(
        query=[{"role": "user", "content": "Move to the glass"}],
        task="trajectory",
        images=[loaded_img],
    )
    # Test general — output should be a string
    result = run_model(
        RoboBrain2,
        inputs=[data_general],
        log_output=True,
    )
    assert "output" in result
    assert "thinking" in result
    assert isinstance(result["output"], str)

    # Test grounding — output should be list of [x1, y1, x2, y2] boxes
    result = run_model(
        RoboBrain2,
        inputs=[data_grounding],
        log_output=True,
    )
    assert "output" in result
    assert isinstance(result["output"], list)

    # Test pointing — output should be list of (x, y) tuples
    result = run_model(
        RoboBrain2,
        inputs=[data_pointing],
        log_output=True,
    )
    assert "output" in result
    assert isinstance(result["output"], list)

    # Test affordance — output should be list of [x1, y1, x2, y2] boxes
    result = run_model(
        RoboBrain2,
        inputs=[data_affordance],
        log_output=True,
    )
    assert "output" in result
    assert isinstance(result["output"], list)

    # Test trajectory — output should be list of point lists
    result = run_model(
        RoboBrain2,
        inputs=[data_trajectory],
        log_output=True,
    )
    assert "output" in result
    assert isinstance(result["output"], list)


def _image_size(img_b64: str) -> tuple[int, int]:
    """(width, height) of a base64 encoded image."""
    img = cv2.imdecode(
        np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR
    )
    return img.shape[1], img.shape[0]


def _inside(width: int, height: int, *points) -> bool:
    return all(0 <= x < width and 0 <= y < height for x, y in points)


def test_planning_outputs_are_pixels_of_the_input_image(loaded_img):
    """Whatever the RoboBrain family, structured outputs come back as pixel
    coordinates of the input image (2.5 predicts on a 0-1000 grid), boxes
    for grounding AND affordance, and 2D trajectory waypoints."""
    width, height = _image_size(loaded_img)
    query = {"role": "user", "content": "The sandwich"}

    boxes = run_model(
        RoboBrain2,
        inputs=[PlanningInput(query=[query], task="grounding", images=[loaded_img])],
        log_output=True,
    )["output"]
    assert boxes, "grounding found nothing on the test image"
    for x1, y1, x2, y2 in boxes:
        assert _inside(width, height, (x1, y1), (x2, y2)) and x1 <= x2 and y1 <= y2

    affordance = run_model(
        RoboBrain2,
        inputs=[
            PlanningInput(
                query=[{"role": "user", "content": "Pick up the sandwich"}],
                task="affordance",
                images=[loaded_img],
            )
        ],
        log_output=True,
    )["output"]
    assert affordance, "affordance found nothing on the test image"
    for box in affordance:
        assert len(box) == 4 and _inside(width, height, box[:2], box[2:])

    points = run_model(
        RoboBrain2,
        inputs=[PlanningInput(query=[query], task="pointing", images=[loaded_img])],
        log_output=True,
    )["output"]
    assert points and _inside(width, height, *points)

    trajectory = run_model(
        RoboBrain2,
        inputs=[
            PlanningInput(
                query=[{"role": "user", "content": "Move to the glass"}],
                task="trajectory",
                images=[loaded_img],
            )
        ],
        log_output=True,
    )
    (waypoints,) = trajectory["output"]
    assert waypoints and all(len(p) == 2 for p in waypoints)
    assert _inside(width, height, *waypoints)
    # the 2.5 family predicts depth alongside; it travels separately
    if "depths" in trajectory:
        (depths,) = trajectory["depths"]
        assert len(depths) == len(waypoints)
