"""CI-safe tests: TransformersTTS unit tests with mocked HuggingFace models."""

import logging
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import torch
import pytest

from roboml.models.text_to_speech import TransformersTTS
from roboml.interfaces import TextToSpeechInput


@pytest.fixture
def mock_waveform_generative():
    """Mock a generative waveform model (like Bark)."""
    model = MagicMock()
    model.can_generate.return_value = True
    model.to.return_value = model
    model.generation_config.sample_rate = 24000
    model.generate.return_value = torch.randn(1, 24000)
    return model


@pytest.fixture
def mock_waveform_forward():
    """Mock a forward-only waveform model (like VITS)."""
    model = MagicMock()
    model.can_generate.return_value = False
    model.to.return_value = model
    model.config.sampling_rate = 16000

    output = MagicMock()
    output.waveform = torch.randn(1, 16000)
    model.return_value = output
    return model


@pytest.fixture
def mock_processor():
    processor = MagicMock()
    proc_output = {"input_ids": torch.tensor([[1, 2, 3]])}
    processor.return_value = proc_output
    return processor


class TestTransformersTTSInit:
    def test_instantiation(self):
        tts = TransformersTTS(logger=logging.getLogger("test"))
        assert tts.model is None
        assert tts.vocoder is None
        assert tts.is_spectrogram_model is False

    @patch("roboml.models.text_to_speech.AutoProcessor")
    @patch("roboml.models.text_to_speech.AutoModelForTextToWaveform")
    def test_initialize_waveform_model(
        self, mock_model_cls, mock_proc_cls, mock_waveform_generative, mock_processor
    ):
        mock_model_cls.from_pretrained.return_value = mock_waveform_generative
        mock_proc_cls.from_pretrained.return_value = mock_processor

        tts = TransformersTTS(logger=logging.getLogger("test"))
        tts._initialize(checkpoint="test/bark")

        assert tts.is_spectrogram_model is False
        assert tts.vocoder is None
        mock_model_cls.from_pretrained.assert_called_once()

    @patch("transformers.SpeechT5HifiGan")
    @patch("roboml.models.text_to_speech.AutoProcessor")
    @patch("roboml.models.text_to_speech.AutoModelForTextToSpectrogram")
    @patch("roboml.models.text_to_speech.AutoModelForTextToWaveform")
    def test_initialize_spectrogram_model_fallback(
        self,
        mock_wav_cls,
        mock_spec_cls,
        mock_proc_cls,
        mock_vocoder_cls,
    ):
        # Waveform loading fails
        mock_wav_cls.from_pretrained.side_effect = ValueError("Not a waveform model")

        # Spectrogram loading succeeds
        spec_model = MagicMock()
        spec_model.to.return_value = spec_model
        spec_model.config.sampling_rate = 16000
        mock_spec_cls.from_pretrained.return_value = spec_model

        mock_proc_cls.from_pretrained.return_value = MagicMock()

        vocoder = MagicMock()
        vocoder.to.return_value = vocoder
        mock_vocoder_cls.from_pretrained.return_value = vocoder

        tts = TransformersTTS(logger=logging.getLogger("test"))
        tts._initialize(checkpoint="microsoft/speecht5_tts")

        assert tts.is_spectrogram_model is True
        assert tts.vocoder is not None


class TestTransformersTTSInference:
    @patch("roboml.models.text_to_speech.AutoProcessor")
    @patch("roboml.models.text_to_speech.AutoModelForTextToWaveform")
    def test_generative_inference(
        self, mock_model_cls, mock_proc_cls, mock_waveform_generative, mock_processor
    ):
        mock_model_cls.from_pretrained.return_value = mock_waveform_generative
        mock_proc_cls.from_pretrained.return_value = mock_processor

        tts = TransformersTTS(logger=logging.getLogger("test"))
        tts._initialize(checkpoint="test/bark")

        data = TextToSpeechInput(query="Hello world", get_bytes=False)
        result = tts._inference(data)

        assert "output" in result
        assert isinstance(result["output"], str)  # base64 encoded audio

    @patch("roboml.models.text_to_speech.AutoProcessor")
    @patch("roboml.models.text_to_speech.AutoModelForTextToWaveform")
    def test_forward_inference(
        self, mock_model_cls, mock_proc_cls, mock_waveform_forward, mock_processor
    ):
        mock_model_cls.from_pretrained.return_value = mock_waveform_forward
        mock_proc_cls.from_pretrained.return_value = mock_processor

        tts = TransformersTTS(logger=logging.getLogger("test"))
        tts._initialize(checkpoint="test/vits")

        data = TextToSpeechInput(query="Hello world", get_bytes=True)
        result = tts._inference(data)

        assert "output" in result
        assert isinstance(result["output"], bytes)

    @patch("roboml.models.text_to_speech.AutoProcessor")
    @patch("roboml.models.text_to_speech.AutoModelForTextToWaveform")
    def test_voice_override(
        self, mock_model_cls, mock_proc_cls, mock_waveform_generative, mock_processor
    ):
        mock_model_cls.from_pretrained.return_value = mock_waveform_generative
        mock_proc_cls.from_pretrained.return_value = mock_processor

        tts = TransformersTTS(logger=logging.getLogger("test"))
        tts._initialize(checkpoint="test/bark", voice="default_voice")

        # Override voice in inference
        data = TextToSpeechInput(query="Hello", voice="custom_voice")
        tts._inference(data)

        # The voice should have been passed to processor
        assert mock_processor.call_count >= 1


class TestSampleRateDetection:
    def test_bark_style_sample_rate(self):
        tts = TransformersTTS(logger=logging.getLogger("test"))
        tts.model = MagicMock()
        tts.model.generation_config.sample_rate = 24000
        assert tts._get_sample_rate() == 24000

    def test_vits_style_sample_rate(self):
        tts = TransformersTTS(logger=logging.getLogger("test"))
        tts.model = MagicMock(spec=[])
        tts.model.config = MagicMock()
        tts.model.config.sampling_rate = 22050
        assert tts._get_sample_rate() == 22050

    def test_fallback_sample_rate(self):
        tts = TransformersTTS(logger=logging.getLogger("test"))
        tts.model = MagicMock(spec=[])
        tts.model.config = MagicMock(spec=[])
        assert tts._get_sample_rate() == 16000
