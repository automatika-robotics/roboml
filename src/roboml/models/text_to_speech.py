from typing import Optional

import torch
from datasets import arrow_dataset, load_dataset
from transformers import (
    AutoProcessor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    BarkModel,
)

from roboml.interfaces import TextToSpeechInput
from roboml.utils import post_process_audio

from ._base import ModelTemplate


class Bark(ModelTemplate):
    """
    Bark TTS model for text to audio (by Suno AI)
    """

    def _initialize(
        self,
        checkpoint: str = "suno/bark-small",
        voice: str = "v2/en_speaker_6",
    ) -> None:
        """
        Initializes the model.
        """
        self.model = BarkModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.pre_processor = AutoProcessor.from_pretrained(checkpoint)
        self.voice = voice
        self.logger.warning(self.model.device)

    def _inference(self, data: TextToSpeechInput) -> dict:
        """Model Inference.
        :param data:
        :type data: TextToSpeechInput
        :rtype: dict
        """
        voice = data.voice or self.voice
        # Speaker embedding loaded along with input
        inputs = self.pre_processor(text=data.query, voice_preset=voice).to(self.device)

        # generate speech
        with torch.no_grad():
            speech = self.model.generate(
                inputs.input_ids, semantic_max_new_tokens=500, do_sample=True
            )
        # get bytes
        sample_rate = self.model.generation_config.sample_rate
        audio = post_process_audio(
            speech.cpu(), sample_rate=sample_rate, get_bytes=data.get_bytes
        )

        return {"output": audio}


class SpeechT5(ModelTemplate):
    """
    SpeechT5 TTS model for text to audio.
    """

    vocoder: Optional[SpeechT5HifiGan] = None
    speaker_dataset: Optional[arrow_dataset.Dataset] = None

    speakers: dict = {
        "awb": 0,  # Scottish male
        "bdl": 1138,  # US male
        "clb": 2271,  # US female
        "jmk": 3403,  # Canadian male
        "ksp": 4535,  # Indian male
        "rms": 5667,  # US male
        "slt": 6799,  # US female
    }
    speaker_dataset_vects: str = "Matthijs/cmu-arctic-xvectors"

    def _initialize(
        self,
        checkpoint: str = "microsoft/speecht5_tts",
        voice: str = "clb",
    ) -> None:
        """
        Initializes the model.
        """
        # Run on GPU with FP16
        self.model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint).to(self.device)
        self.pre_processor = AutoProcessor.from_pretrained(checkpoint)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(
            self.device
        )
        # Load speaker embeddings (consider caching this locally if network is slow)
        try:
            self.speaker_dataset = load_dataset(
                self.speaker_dataset_vects, split="validation"
            )
        except Exception as e:
            self.logger.error(f"Failed to load speaker dataset: {e}")
            self.speaker_dataset = None  # Handle gracefully in inference

        self.voice = voice

    def _inference(self, data: TextToSpeechInput) -> dict:
        """Model Inference.
        :param data:
        :type data: TextToSpeechInput
        :rtype: dict
        """
        inputs = self.pre_processor(text=data.query, return_tensors="pt").to(
            self.device
        )

        voice = data.voice or self.voice

        # load speaker embeddings
        speaker_embeddings = self._get_speaker_embedding(voice)

        if speaker_embeddings is None:
            raise ValueError(f"Failed to load speaker embedding for voice '{voice}'.")

        # generate speech with the model
        with torch.no_grad():
            speech: torch.FloatTensor = self.model.generate_speech(
                inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder
            )

        # get bytes
        audio = post_process_audio(speech.cpu(), get_bytes=data.get_bytes)

        return {"output": audio}

    def _get_speaker_embedding(self, speaker_id: str) -> Optional[torch.Tensor]:
        """Safely retrieves speaker embedding tensor."""
        if self.speaker_dataset is None:
            self.logger.error(
                "Speaker dataset not loaded. Cannot provide speaker embeddings."
            )
            return None
        try:
            speaker_idx = self.speakers[speaker_id]
            embedding = torch.tensor(
                self.speaker_dataset[speaker_idx]["xvector"], dtype=torch.float32
            )  # Embeddings usually float32
            return embedding.unsqueeze(0).to(
                self.device
            )  # Add batch dim and move to device
        except KeyError:
            self.logger.warning(f"Speaker ID '{speaker_id}' not found in speaker map.")
            return None
        except IndexError:
            self.logger.warning(
                f"Speaker index for '{speaker_id}' out of bounds for the loaded dataset."
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Error loading speaker embedding for '{speaker_id}': {e}"
            )
            return None


class MeloTTS(ModelTemplate):
    """
    MeloTTS model for text to audio (by MyShell AI).
    Uses the official melo.api.TTS class.
    """

    # Based on common MeloTTS models
    supported_languages: list = ["EN", "ES", "FR", "ZH", "JP", "KR"]

    def _initialize(
        self,
        language: str = "EN",  # Default language
        speaker_id: str = "EN-US",  # Default speaker for English
    ) -> None:
        """
        Initialize model. Downloads models automatically on first use if needed via load_or_download functions used internally by MeloTTS.
        """
        try:
            # The TTS class from api.py is typically imported as MeloTTS via melo.api
            from melo.api import TTS as MeloTTSModel

        except ModuleNotFoundError as e:
            self.logger.error(
                """MeloTTS library is not install. It can be installed as follows:
                              - pip install git+https://github.com/myshell-ai/MeloTTS.git
                              - python -m unidic download
                              """,
                exc_info=True,
            )
            raise ModuleNotFoundError("""MeloTTS library is not install. It can be installed as follows:
                              - pip install git+https://github.com/myshell-ai/MeloTTS.git
                              - python -m unidic download""") from e

        if language not in self.supported_languages:
            self.logger.warning(
                f"Language '{language}' might not be officially supported by this wrapper. Supported: {self.supported_languages}. Trying anyway."
            )

        self.language = language  # Store language used for init
        self.default_speaker_key = speaker_id  # Store the key (e.g., "EN-US")

        self.logger.info(
            f"Initializing MeloTTS for language '{language}' on device '{self.device}'..."
        )
        # Instantiate the TTS class from melo.api using parameters from api.py
        self.model = MeloTTSModel(language=language, device=self.device)
        self.speaker_ids = (
            self.model.hps.data.spk2id
        )  # Get speaker map from model's hps
        self.logger.info(
            f"Available speakers for {language}: {list(self.speaker_ids.keys())}"
        )

        # Validate default speaker ID
        if self.default_speaker_key not in self.speaker_ids:
            available_speakers = list(self.speaker_ids.keys())
            fallback_speaker = available_speakers[0] if available_speakers else None
            self.logger.warning(
                f"Default speaker '{self.default_speaker_key}' not found for language '{self.language}'. "
                f"Available: {available_speakers}. Falling back to '{fallback_speaker}'."
            )
            self.default_speaker_key = fallback_speaker
        else:
            self.logger.info(f"Default speaker set to: {self.default_speaker_key}")

        if self.default_speaker_key is None:
            raise ValueError(
                f"No speakers available for language '{self.language}'. Cannot initialize."
            )

    def _inference(self, data: TextToSpeechInput) -> dict:
        """Model Inference using MeloTTS.
        Generates audio directly as a NumPy array by calling tts_to_file with
        output_path=None, as supported by the API.

        :param data: Input data containing text query and optional voice (speaker key).
        :type data: TextToSpeechInput
        :rtype: dict containing the audio output
        """

        text = data.query
        # Use provided voice (speaker key) or the default for the initialized language
        speaker_key = data.voice or self.default_speaker_key
        self.logger.info(f"Using speaker key {speaker_key}")

        # Get the internal speaker ID number expected by the model
        sid = self.speaker_ids[speaker_key]
        self.logger.info(f"Speaker ID {sid}")

        try:
            # Call tts_to_file with output_path=None to get audio directly
            audio_np = self.model.tts_to_file(
                text,
                sid,
                output_path=None,  # Get numpy array directly
                speed=1.0,
                quiet=True,
            )

            sample_rate = (
                self.model.hps.data.sampling_rate
            )  # Get sample rate from model's hps

            # Post-process the audio tensor
            output = post_process_audio(
                audio_np,
                sample_rate=sample_rate,
                get_bytes=data.get_bytes,
            )
            return {"output": output}

        except Exception as e:
            self.logger.error(f"Error during MeloTTS inference: {e}", exc_info=True)
            if "cuda" in self.device:
                torch.cuda.empty_cache()
            raise
