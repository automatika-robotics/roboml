"""CI-safe tests: validate Pydantic interface models."""

import numpy as np
import pytest
from pydantic import ValidationError


def test_detection_input_valid():
    """Test DetectionInput with valid base64 string images."""
    from roboml.interfaces import DetectionInput

    data = DetectionInput(
        images=["aGVsbG8="],  # base64 "hello"
        threshold=0.7,
        get_dataset_labels=True,
    )
    assert data.threshold == 0.7
    assert data.get_dataset_labels is True
    assert data.labels_to_track is None


def test_detection_input_with_numpy():
    """Test DetectionInput with numpy array images."""
    from roboml.interfaces import DetectionInput

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    data = DetectionInput(images=[img], threshold=0.3)
    assert data.threshold == 0.3
    assert isinstance(data.images[0], np.ndarray)


def test_detection_input_with_tracking():
    """Test DetectionInput with tracking labels."""
    from roboml.interfaces import DetectionInput

    data = DetectionInput(
        images=["aGVsbG8="],
        threshold=0.5,
        labels_to_track=["person", "car"],
    )
    assert data.labels_to_track == ["person", "car"]


def test_detection_input_empty_images_rejected():
    """Test that DetectionInput rejects empty image list."""
    from roboml.interfaces import DetectionInput

    with pytest.raises(ValidationError):
        DetectionInput(images=[], threshold=0.5)


def test_detection_input_defaults():
    """Test DetectionInput default values."""
    from roboml.interfaces import DetectionInput

    data = DetectionInput(images=["aGVsbG8="])
    assert data.threshold == 0.5
    assert data.get_dataset_labels is True
    assert data.labels_to_track is None


def test_llm_input_valid():
    """Test LLMInput with valid messages."""
    from roboml.interfaces import LLMInput

    data = LLMInput(
        query=[{"role": "user", "content": "Hello"}],
        max_new_tokens=50,
        temperature=0.5,
    )
    assert data.max_new_tokens == 50
    assert data.stream is False


def test_llm_input_empty_query_rejected():
    """Test that LLMInput rejects empty query."""
    from roboml.interfaces import LLMInput

    with pytest.raises(ValidationError):
        LLMInput(query=[])


def test_vllm_input_valid():
    """Test VLLMInput with images."""
    from roboml.interfaces import VLLMInput

    data = VLLMInput(
        query=[{"role": "user", "content": "What is this?"}],
        images=["aGVsbG8="],
    )
    assert len(data.images) == 1


def test_vllm_input_video_only():
    """Test VLLMInput with a video and no images."""
    from roboml.interfaces import VLLMInput

    data = VLLMInput(
        query=[{"role": "user", "content": "What happens in this video?"}],
        videos=["aGVsbG8="],
        video_fps=4.0,
        max_video_frames=16,
    )
    assert data.images == []
    assert len(data.videos) == 1
    assert data.video_fps == 4.0
    assert data.max_video_frames == 16


def test_vllm_input_video_as_frames():
    """Test VLLMInput with a video as a numpy frame array."""
    from roboml.interfaces import VLLMInput

    frames = np.zeros((8, 64, 64, 3), dtype=np.uint8)
    data = VLLMInput(
        query=[{"role": "user", "content": "Describe"}],
        videos=[frames],
    )
    assert isinstance(data.videos[0], np.ndarray)
    assert data.video_fps is None


def test_vllm_input_images_and_videos():
    """Test VLLMInput with both images and videos."""
    from roboml.interfaces import VLLMInput

    data = VLLMInput(
        query=[{"role": "user", "content": "Compare"}],
        images=["aGVsbG8="],
        videos=["d29ybGQ="],
    )
    assert len(data.images) == 1
    assert len(data.videos) == 1


def test_vllm_input_single_frame_video_rejected():
    """Test that a bare (H, W, C) image passed as a video is rejected."""
    from roboml.interfaces import VLLMInput

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValidationError, match=r"\(T, H, W, C\)"):
        VLLMInput(
            query=[{"role": "user", "content": "Describe"}],
            videos=[image],
            video_fps=4.0,
        )


def test_vllm_input_non_rgb_video_rejected():
    """Test that frame stacks without 3 channels are rejected."""
    from roboml.interfaces import VLLMInput

    rgba = np.zeros((8, 64, 64, 4), dtype=np.uint8)
    with pytest.raises(ValidationError, match="C=3"):
        VLLMInput(query=[{"role": "user", "content": "Describe"}], videos=[rgba])


def test_vllm_input_no_visual_input_rejected():
    """Test that VLLMInput rejects requests without images or videos."""
    from roboml.interfaces import VLLMInput

    with pytest.raises(ValidationError):
        VLLMInput(query=[{"role": "user", "content": "Hello"}])

    with pytest.raises(ValidationError):
        VLLMInput(query=[{"role": "user", "content": "Hello"}], images=[], videos=[])


def test_llm_input_default_max_new_tokens():
    """Test the LLMInput max_new_tokens default."""
    from roboml.interfaces import LLMInput

    data = LLMInput(query=[{"role": "user", "content": "Hello"}])
    assert data.max_new_tokens == 512


def test_planning_input_validation():
    """Test PlanningInput task validation."""
    from roboml.interfaces import PlanningInput

    # general task allows multiple images
    data = PlanningInput(
        query=[{"role": "user", "content": "Describe"}],
        images=["aGVsbG8=", "d29ybGQ="],
        task="general",
    )
    assert data.task == "general"

    # pointing task requires exactly one image
    with pytest.raises(ValidationError):
        PlanningInput(
            query=[{"role": "user", "content": "Point"}],
            images=["aGVsbG8=", "d29ybGQ="],
            task="pointing",
        )

    # planning models require images
    with pytest.raises(ValidationError):
        PlanningInput(query=[{"role": "user", "content": "Describe"}])

    # planning models do not accept videos
    with pytest.raises(ValidationError):
        PlanningInput(
            query=[{"role": "user", "content": "Describe"}],
            images=["aGVsbG8="],
            videos=["d29ybGQ="],
        )


def test_tts_input_valid():
    """Test TextToSpeechInput."""
    from roboml.interfaces import TextToSpeechInput

    data = TextToSpeechInput(query="Hello world")
    assert data.voice is None
    assert data.get_bytes is False


def test_stt_input_valid():
    """Test SpeechToTextInput."""
    from roboml.interfaces import SpeechToTextInput

    data = SpeechToTextInput(query="aGVsbG8=")
    assert data.max_new_tokens is None
    assert data.vad_filter is False
