from typing import Optional

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForTextToWaveform,
    AutoModelForTextToSpectrogram,
)

from roboml.interfaces import TextToSpeechInput
from roboml.utils import post_process_audio

from ._base import ModelTemplate


class TransformersTTS(ModelTemplate):
    """
    Generic text-to-speech model from HuggingFace Transformers.
    Supports all models registered under AutoModelForTextToWaveform
    (Bark, VITS, SeamlessM4T, MusicGen, etc.) and AutoModelForTextToSpectrogram
    (SpeechT5 with vocoder).
    """

    def __init__(self, **kwargs):
        """__init__.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.vocoder = None
        self.is_spectrogram_model: bool = False
        self.voice: Optional[str] = None
        self.speaker_embeddings: Optional[torch.Tensor] = None

    def _initialize(
        self,
        checkpoint: str = "suno/bark-small",
        voice: Optional[str] = "v2/en_speaker_6",
        vocoder_checkpoint: Optional[str] = None,
    ) -> None:
        """Initialize TTS model.
        :param checkpoint: HuggingFace model ID
        :type checkpoint: str
        :param voice: Voice preset (Bark) or speaker ID. Meaning is model-specific.
        :type voice: Optional[str]
        :param vocoder_checkpoint: Vocoder model ID for spectrogram models (e.g. SpeechT5).
            If not provided and a spectrogram model is loaded, defaults to
            'microsoft/speecht5_hifigan'.
        :type vocoder_checkpoint: Optional[str]
        :rtype: None
        """
        self.voice = voice

        # Try loading as waveform model first, fall back to spectrogram
        try:
            self.model = AutoModelForTextToWaveform.from_pretrained(
                checkpoint, dtype=torch.float16
            ).to(self.device)
            self.is_spectrogram_model = False
            self.logger.info(f"Loaded waveform model: {checkpoint}")
        except (ValueError, OSError):
            self.model = AutoModelForTextToSpectrogram.from_pretrained(checkpoint).to(
                self.device
            )
            self.is_spectrogram_model = True
            self.logger.info(f"Loaded spectrogram model: {checkpoint}")

            # Load vocoder for spectrogram models
            vocoder_id = vocoder_checkpoint or "microsoft/speecht5_hifigan"
            from transformers import SpeechT5HifiGan

            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_id).to(self.device)
            self.logger.info(f"Loaded vocoder: {vocoder_id}")

        # Load processor/tokenizer
        try:
            self.pre_processor = AutoProcessor.from_pretrained(checkpoint)
        except (OSError, KeyError):
            self.pre_processor = AutoTokenizer.from_pretrained(checkpoint)

    def _inference(self, data: TextToSpeechInput) -> dict:
        """Model Inference.
        :param data:
        :type data: TextToSpeechInput
        :rtype: dict
        """
        voice = data.voice or self.voice

        if self.model.can_generate():
            audio_array, sample_rate = self._generate_inference(data.query, voice)
        else:
            audio_array, sample_rate = self._forward_inference(data.query)

        audio = post_process_audio(
            audio_array, sample_rate=sample_rate, get_bytes=data.get_bytes
        )
        return {"output": audio}

    def _generate_inference(self, text: str, voice: Optional[str]) -> tuple:
        """Inference for generative models (Bark, SpeechT5, SeamlessM4T, etc.)."""
        proc_kwargs = {"text": text, "return_tensors": "pt"}

        # Try passing voice_preset (Bark-specific), fall back without it
        if voice:
            try:
                inputs = self.pre_processor(**proc_kwargs, voice_preset=voice)
            except TypeError:
                inputs = self.pre_processor(**proc_kwargs)
        else:
            inputs = self.pre_processor(**proc_kwargs)

        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}

        # Build generate kwargs
        gen_kwargs = dict(inputs)
        if self.is_spectrogram_model and self.vocoder:
            gen_kwargs["vocoder"] = self.vocoder
            # SpeechT5 requires speaker embeddings
            if self.speaker_embeddings is None:
                # Use zero embeddings as default
                self.speaker_embeddings = torch.zeros((1, 512)).to(self.device)
            gen_kwargs["speaker_embeddings"] = self.speaker_embeddings

        with torch.no_grad():
            speech = self.model.generate(**gen_kwargs)

        audio_array = speech.cpu().float().numpy().squeeze()
        sample_rate = self._get_sample_rate()

        return audio_array, sample_rate

    def _forward_inference(self, text: str) -> tuple:
        """Inference for forward-only models (VITS, etc.)."""
        inputs = self.pre_processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}

        with torch.no_grad():
            outputs = self.model(**inputs)

        audio_array = outputs.waveform.cpu().float().numpy().squeeze()
        sample_rate = self._get_sample_rate()

        return audio_array, sample_rate

    def _get_sample_rate(self) -> int:
        """Get sample rate from model config or generation config."""
        # Bark stores sample_rate in generation_config
        if hasattr(self.model, "generation_config") and hasattr(
            self.model.generation_config, "sample_rate"
        ):
            return self.model.generation_config.sample_rate

        # VITS, SpeechT5 store sampling_rate in config
        if hasattr(self.model.config, "sampling_rate"):
            return self.model.config.sampling_rate

        # Fallback
        self.logger.warning("Could not determine sample rate, defaulting to 16000")
        return 16000
