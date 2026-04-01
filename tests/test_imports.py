"""CI-safe tests: verify all imports, module exports, and package structure."""

import pytest


def test_import_roboml():
    """Test that the roboml package can be imported."""
    import roboml

    assert roboml is not None


def test_import_models_module():
    """Test that all model classes can be imported from roboml.models."""
    from roboml.models import (
        ModelTemplate,
        TransformersLLM,
        TransformersMLLM,
        Whisper,
        TransformersTTS,
        RoboBrain2,
        VisionModel,
    )

    assert ModelTemplate is not None
    assert VisionModel is not None
    assert TransformersTTS is not None


def test_import_interfaces():
    """Test that all interface classes can be imported."""
    from roboml.interfaces import (
        LLMInput,
        VLLMInput,
        PlanningInput,
        DetectionInput,
        SpeechToTextInput,
        TextToSpeechInput,
    )

    assert DetectionInput is not None


def test_import_utils():
    """Test that utility functions can be imported."""
    from roboml.utils import (
        pre_process_images_to_pil,
        pre_process_images_to_np,
        b64_str_to_bytes,
        post_process_audio,
        get_quantization_config,
        Quantization,
        Status,
        background_task,
        is_background_task,
    )

    assert Status.READY is not None
    assert Quantization.FOUR.value == "4bit"


def test_import_ray_server():
    """Test that ray server components can be imported."""
    from roboml.ray.app_factory import AppFactory
    from roboml.ray.node import RayNode

    assert AppFactory is not None
    assert RayNode is not None


def test_import_resp_server():
    """Test that RESP server components can be imported."""
    from roboml.resp_server.server import Server
    from roboml.resp_server.node import RESPNode

    assert Server is not None
    assert RESPNode is not None


def test_import_trackers():
    """Test that trackers library is available (core dependency)."""
    from trackers import ByteTrackTracker
    import supervision as sv

    assert ByteTrackTracker is not None
    assert sv.Detections is not None


def test_import_transformers_detection():
    """Test that HuggingFace detection model classes are available."""
    from transformers import AutoModelForObjectDetection, AutoImageProcessor

    assert AutoModelForObjectDetection is not None
    assert AutoImageProcessor is not None


def test_vision_model_is_model_template():
    """Test that VisionModel inherits from ModelTemplate."""
    from roboml.models import VisionModel, ModelTemplate

    assert issubclass(VisionModel, ModelTemplate)


def test_no_mmdet_imports():
    """Test that mmdetection is NOT imported anywhere in roboml."""
    import sys
    import roboml.models.vision
    import roboml.utils
    import roboml.ray.app_factory
    import roboml.resp_server.server

    mmdet_modules = [m for m in sys.modules if m.startswith("mmdet")]
    mmcv_modules = [m for m in sys.modules if m.startswith("mmcv")]
    mmengine_modules = [m for m in sys.modules if m.startswith("mmengine")]

    assert len(mmdet_modules) == 0, f"mmdet modules found: {mmdet_modules}"
    assert len(mmcv_modules) == 0, f"mmcv modules found: {mmcv_modules}"
    assert len(mmengine_modules) == 0, f"mmengine modules found: {mmengine_modules}"


def test_no_norfair_imports():
    """Test that norfair is NOT imported in roboml."""
    import sys
    import roboml.models.vision

    norfair_modules = [m for m in sys.modules if m.startswith("norfair")]
    assert len(norfair_modules) == 0, f"norfair modules found: {norfair_modules}"
