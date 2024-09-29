from typing import Optional

import torch
from datasets import arrow_dataset, load_dataset
from transformers import (
    AutoProcessor,
    BarkModel,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)

from roboml.interfaces import TextToSpeechInput
from roboml.ray import app, ingress_decorator
from roboml.utils import post_process_audio

from ._base import ModelTemplate


@ingress_decorator
class Bark(ModelTemplate):
    """
    Bark TTS model for text to audio (by Suno AI)
    """

    @app.post("/initialize")
    def _initialize(
        self,
        checkpoint: str = "suno/bark-small",
        attn_implementation: Optional[str] = "flash_attention_2",
        voice: str = "v2/en_speaker_6",
    ) -> None:
        """
        Initializes the model.
        """
        self.model = BarkModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            attn_implementation=attn_implementation,
        ).to(self.device)
        self.pre_processor = AutoProcessor.from_pretrained(checkpoint)
        self.voice = voice

    @app.post("/inference")
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
        speech = self.model.generate(**inputs, semantic_max_new_tokens=500)
        # get bytes
        sample_rate = self.model.generation_config.sample_rate
        audio = post_process_audio(
            speech, sample_rate=sample_rate, get_bytes=data.get_bytes
        )

        return {"output": audio}


@ingress_decorator
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

    @app.post("/initialize")
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
        self.speaker_dataset = load_dataset(
            self.speaker_dataset_vects, split="validation"
        )
        self.voice = voice

    @app.post("/inference")
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
        speaker_embeddings: torch.FloatTensor = (
            torch.tensor(self.speaker_dataset[self.speakers[voice]]["xvector"])
            .unsqueeze(0)
            .to(self.device)
        )

        # generate speech with the model
        speech: torch.FloatTensor = self.model.generate_speech(
            inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder
        )

        # get bytes
        audio = post_process_audio(speech, get_bytes=data.get_bytes)

        return {"output": audio}
