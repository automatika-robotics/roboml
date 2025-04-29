import numpy as np
from faster_whisper import WhisperModel

from roboml.interfaces import SpeechToTextInput

from ._base import ModelTemplate


class Whisper(ModelTemplate):
    """
    Whisper model for Audio to text.
    """

    def _initialize(
        self,
        checkpoint: str = "small.en",
        quantization: str = "int8",
    ) -> None:
        """
        Initializes the model.
        """
        # Run on GPU with FP16
        self.model = WhisperModel(
            checkpoint,
            device=self.device,
            compute_type=quantization,
        )

    def _inference(self, data: SpeechToTextInput) -> dict:
        """
        Model Inference
        :param      data:           Model Input
        :type       data:           SpeechToTextInput
        :returns:   Model output
        :rtype:     dict
        """
        # make inference
        segments, _ = self.model.transcribe(
            audio=data.query, max_new_tokens=data.max_new_tokens
        )
        return {"output": " ".join([s.text.strip() for s in segments])}
