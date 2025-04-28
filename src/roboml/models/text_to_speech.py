from typing import Optional

import torch
from datasets import arrow_dataset, load_dataset
from transformers import (
    AutoProcessor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)

from roboml.interfaces import TextToSpeechInput
from roboml.utils import post_process_audio

from ._base import ModelTemplate


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
        self.speaker_dataset = load_dataset(
            self.speaker_dataset_vects, split="validation"
        )
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
