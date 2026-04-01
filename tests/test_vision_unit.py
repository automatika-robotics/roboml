"""CI-safe tests: VisionModel unit tests with mocked HuggingFace models."""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch
import logging

import numpy as np
import torch
import pytest
from PIL import Image

from roboml.models.vision import VisionModel
from roboml.interfaces import DetectionInput


@pytest.fixture
def sample_b64_image():
    """Create a small test image as base64 string."""
    img = Image.fromarray(np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@pytest.fixture
def mock_hf_model():
    """Mock HuggingFace detection model and processor."""
    model = MagicMock()
    model.config.id2label = {0: "person", 1: "car", 2: "dog"}
    model.to.return_value = model

    # mock model forward pass output
    model_output = MagicMock()
    model.return_value = model_output

    processor = MagicMock()

    # mock processor call (preprocessing)
    proc_output = MagicMock()
    proc_output.to.return_value = proc_output
    proc_output.keys.return_value = ["pixel_values"]
    proc_output.__getitem__ = lambda self, key: torch.zeros(1, 3, 640, 640)
    processor.return_value = proc_output

    # mock post_process_object_detection
    processor.post_process_object_detection.return_value = [
        {
            "scores": torch.tensor([0.95, 0.87, 0.72]),
            "labels": torch.tensor([0, 1, 2]),
            "boxes": torch.tensor([
                [10.0, 20.0, 100.0, 200.0],
                [50.0, 60.0, 150.0, 250.0],
                [200.0, 100.0, 300.0, 350.0],
            ]),
        }
    ]

    return model, processor, model_output


class TestVisionModelInit:
    def test_instantiation(self):
        vm = VisionModel(logger=logging.getLogger("test"))
        assert vm.trackers is None
        assert vm.model is None

    @patch("roboml.models.vision.AutoModelForObjectDetection")
    @patch("roboml.models.vision.AutoImageProcessor")
    def test_initialize_loads_model(self, mock_proc_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = (
            mock_model_cls.from_pretrained.return_value
        )
        mock_model_cls.from_pretrained.return_value.config.id2label = {0: "cat"}
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        vm = VisionModel(logger=logging.getLogger("test"))
        vm._initialize(checkpoint="test/model")

        mock_model_cls.from_pretrained.assert_called_once_with("test/model")
        mock_proc_cls.from_pretrained.assert_called_once_with("test/model")
        assert vm.data_classes == {0: "cat"}

    @patch("roboml.models.vision.AutoModelForObjectDetection")
    @patch("roboml.models.vision.AutoImageProcessor")
    def test_initialize_with_trackers(self, mock_proc_cls, mock_model_cls):
        mock_model_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value.to.return_value = (
            mock_model_cls.from_pretrained.return_value
        )
        mock_model_cls.from_pretrained.return_value.config.id2label = {0: "cat"}
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        vm = VisionModel(logger=logging.getLogger("test"))
        vm._initialize(checkpoint="test/model", setup_trackers=True, num_trackers=2)

        assert vm.trackers is not None
        assert len(vm.trackers) == 2


class TestVisionModelInference:
    @patch("roboml.models.vision.AutoModelForObjectDetection")
    @patch("roboml.models.vision.AutoImageProcessor")
    def test_inference_output_format(
        self, mock_proc_cls, mock_model_cls, mock_hf_model, sample_b64_image
    ):
        model, processor, _ = mock_hf_model
        mock_model_cls.from_pretrained.return_value = model
        mock_proc_cls.from_pretrained.return_value = processor

        vm = VisionModel(logger=logging.getLogger("test"))
        vm._initialize(checkpoint="test/model")

        data = DetectionInput(images=[sample_b64_image], threshold=0.5)
        result = vm._inference(data)

        assert "output" in result
        assert isinstance(result["output"], list)
        assert len(result["output"]) == 1

        det = result["output"][0]
        assert "bboxes" in det
        assert "labels" in det
        assert "scores" in det
        assert len(det["bboxes"]) == 3
        assert len(det["labels"]) == 3
        assert len(det["scores"]) == 3

    @patch("roboml.models.vision.AutoModelForObjectDetection")
    @patch("roboml.models.vision.AutoImageProcessor")
    def test_inference_with_dataset_labels(
        self, mock_proc_cls, mock_model_cls, mock_hf_model, sample_b64_image
    ):
        model, processor, _ = mock_hf_model
        mock_model_cls.from_pretrained.return_value = model
        mock_proc_cls.from_pretrained.return_value = processor

        vm = VisionModel(logger=logging.getLogger("test"))
        vm._initialize(checkpoint="test/model")

        data = DetectionInput(
            images=[sample_b64_image], threshold=0.5, get_dataset_labels=True
        )
        result = vm._inference(data)

        labels = result["output"][0]["labels"]
        assert labels == ["person", "car", "dog"]

    @patch("roboml.models.vision.AutoModelForObjectDetection")
    @patch("roboml.models.vision.AutoImageProcessor")
    def test_inference_without_dataset_labels(
        self, mock_proc_cls, mock_model_cls, mock_hf_model, sample_b64_image
    ):
        model, processor, _ = mock_hf_model
        mock_model_cls.from_pretrained.return_value = model
        mock_proc_cls.from_pretrained.return_value = processor

        vm = VisionModel(logger=logging.getLogger("test"))
        vm._initialize(checkpoint="test/model")

        data = DetectionInput(
            images=[sample_b64_image], threshold=0.5, get_dataset_labels=False
        )
        result = vm._inference(data)

        labels = result["output"][0]["labels"]
        # should be integer labels
        assert labels == [0, 1, 2]

    @patch("roboml.models.vision.AutoModelForObjectDetection")
    @patch("roboml.models.vision.AutoImageProcessor")
    def test_inference_empty_detections(
        self, mock_proc_cls, mock_model_cls, sample_b64_image
    ):
        model = MagicMock()
        model.config.id2label = {0: "person"}
        model.to.return_value = model

        processor = MagicMock()
        proc_output = MagicMock()
        proc_output.to.return_value = proc_output
        processor.return_value = proc_output
        processor.post_process_object_detection.return_value = [
            {
                "scores": torch.tensor([]),
                "labels": torch.tensor([], dtype=torch.long),
                "boxes": torch.zeros((0, 4)),
            }
        ]

        mock_model_cls.from_pretrained.return_value = model
        mock_proc_cls.from_pretrained.return_value = processor

        vm = VisionModel(logger=logging.getLogger("test"))
        vm._initialize(checkpoint="test/model")

        data = DetectionInput(images=[sample_b64_image], threshold=0.9)
        result = vm._inference(data)

        assert result["output"][0] == {}


class TestVisionModelFilter:
    def test_filter_by_threshold(self):
        vm = VisionModel(logger=logging.getLogger("test"))
        scores = np.array([0.9, 0.3, 0.7, 0.1])
        labels = np.array([0, 1, 2, 3])
        bboxes = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ])

        scores_f, labels_f, _ = vm._filter(0.5, scores, labels, bboxes)
        assert len(scores_f) == 2
        assert list(labels_f) == [0, 2]

    def test_filter_by_labels(self):
        vm = VisionModel(logger=logging.getLogger("test"))
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array(["person", "car", "person"])
        bboxes = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

        scores_f, labels_f, _ = vm._filter(np.array(["person"]), scores, labels, bboxes)
        assert len(scores_f) == 2
        assert list(labels_f) == ["person", "person"]
