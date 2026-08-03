"""CI-safe tests: validate utility functions."""

import base64

import numpy as np
import pytest

from roboml.utils import (
    Status,
    Quantization,
    pre_process_images_to_pil,
    pre_process_images_to_np,
    pre_process_videos,
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


class TestVideoProcessing:
    @pytest.fixture
    def sample_frames(self):
        return np.random.randint(0, 255, (12, 64, 64, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_mp4_bytes(self, sample_frames):
        """Encode sample frames as an in-memory mp4 at 4 FPS."""
        from io import BytesIO

        import av

        buf = BytesIO()
        with av.open(buf, mode="w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=4)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            for arr in sample_frames:
                frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return buf.getvalue()

    def test_from_ndarray_with_fps(self, sample_frames):
        frames_list, metadata = pre_process_videos([sample_frames], video_fps=4.0)
        assert len(frames_list) == 1
        assert frames_list[0].shape == (12, 64, 64, 3)
        assert metadata is not None
        assert metadata[0]["fps"] == 4.0
        assert metadata[0]["total_num_frames"] == 12
        assert metadata[0]["duration"] == pytest.approx(3.0)
        assert metadata[0]["frames_indices"] == list(range(12))

    def test_from_ndarray_without_fps(self, sample_frames):
        frames_list, metadata = pre_process_videos([sample_frames])
        assert len(frames_list) == 1
        assert metadata is None

    def test_from_encoded_bytes(self, sample_mp4_bytes):
        frames_list, metadata = pre_process_videos([sample_mp4_bytes])
        assert len(frames_list) == 1
        assert frames_list[0].shape[0] == 12
        assert frames_list[0].shape[-1] == 3
        assert metadata is not None
        assert metadata[0]["fps"] == pytest.approx(4.0)

    def test_from_b64_string(self, sample_mp4_bytes):
        b64_video = base64.b64encode(sample_mp4_bytes).decode("utf-8")
        frames_list, metadata = pre_process_videos([b64_video])
        assert len(frames_list) == 1
        assert frames_list[0].shape[0] == 12
        assert metadata is not None

    def test_max_video_frames_subsampling(self, sample_frames):
        frames_list, metadata = pre_process_videos(
            [sample_frames], video_fps=4.0, max_video_frames=6
        )
        assert frames_list[0].shape == (6, 64, 64, 3)
        assert metadata is not None
        assert metadata[0]["total_num_frames"] == 6
        # fps is scaled down so that duration stays consistent
        assert metadata[0]["fps"] == pytest.approx(2.0)
        assert metadata[0]["duration"] == pytest.approx(3.0)
        # indices refer to the kept frames at the effective fps
        assert metadata[0]["frames_indices"] == list(range(6))

    def test_max_video_frames_applied_during_decode(self, sample_mp4_bytes):
        frames_list, metadata = pre_process_videos(
            [sample_mp4_bytes], max_video_frames=6
        )
        assert frames_list[0].shape == (6, 64, 64, 3)
        assert metadata is not None
        assert metadata[0]["total_num_frames"] == 6
        # fps is scaled down so that duration stays consistent
        assert metadata[0]["fps"] == pytest.approx(2.0)
        assert metadata[0]["duration"] == pytest.approx(3.0)
        assert metadata[0]["frames_indices"] == list(range(6))

    def test_max_video_frames_without_container_index(self, sample_frames):
        """Containers without a frame-count index (e.g. mpegts) must still
        be capped via the duration estimate or the counting fallback."""
        from io import BytesIO

        import av

        buf = BytesIO()
        with av.open(buf, mode="w", format="mpegts") as container:
            stream = container.add_stream("mpeg4", rate=4)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            for arr in sample_frames:
                frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        frames_list, _ = pre_process_videos([buf.getvalue()], max_video_frames=6)
        assert 1 <= frames_list[0].shape[0] <= 6

    def test_invalid_video_bytes_rejected(self):
        # PyAV raises FFmpegError (an OSError subclass) on undecodable bytes
        with pytest.raises((ValueError, OSError)):
            pre_process_videos([b"not a video"])
