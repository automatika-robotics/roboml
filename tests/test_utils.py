"""CI-safe tests: validate utility functions."""

import base64

import numpy as np
import pytest

from roboml.utils import (
    Status,
    Quantization,
    pre_process_images_to_pil,
    pre_process_images_to_np,
    b64_str_to_bytes,
    post_process_audio,
    background_task,
    is_background_task,
)


class TestStatus:
    def test_status_values(self):
        assert Status.LOADED.value == 1
        assert Status.INITIALIZING.value == 2
        assert Status.READY.value == 3
        assert Status.INITIALIZATION_ERROR.value == 4


class TestQuantization:
    def test_quantization_values(self):
        assert Quantization.FOUR.value == "4bit"
        assert Quantization.EIGHT.value == "8bit"


class TestImageProcessing:
    @pytest.fixture
    def sample_ndarray(self):
        return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_b64(self, sample_ndarray):
        from PIL import Image
        from io import BytesIO

        img = Image.fromarray(sample_ndarray)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def test_pil_from_ndarray_list(self, sample_ndarray):
        result = pre_process_images_to_pil([sample_ndarray])
        assert len(result) == 1
        assert result[0].size == (64, 64)

    def test_pil_from_b64_list(self, sample_b64):
        result = pre_process_images_to_pil([sample_b64])
        assert len(result) == 1

    def test_pil_from_ndarray_concatenate(self, sample_ndarray):
        result = pre_process_images_to_pil([sample_ndarray], concatenate=True)
        assert result.size == (64, 64)

    def test_np_from_ndarray_list(self, sample_ndarray):
        result = pre_process_images_to_np([sample_ndarray])
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)

    def test_np_from_b64_list(self, sample_b64):
        result = pre_process_images_to_np([sample_b64])
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)


class TestB64StrToBytes:
    def test_decode(self):
        original = b"hello world"
        encoded = base64.b64encode(original).decode("utf-8")
        result = b64_str_to_bytes(encoded)
        assert result == original


class TestPostProcessAudio:
    def test_returns_base64_string(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = post_process_audio(audio, sample_rate=16000, get_bytes=False)
        assert isinstance(result, str)
        # should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_returns_bytes(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = post_process_audio(audio, sample_rate=16000, get_bytes=True)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestBackgroundTask:
    def test_decorator(self):
        @background_task
        def my_func():
            return 42

        assert my_func() == 42

    def test_is_background_task(self):
        @background_task
        def decorated():
            pass

        def not_decorated():
            pass

        assert is_background_task(decorated) is True
        assert is_background_task(not_decorated) is False
