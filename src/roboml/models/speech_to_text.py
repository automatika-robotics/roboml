import base64
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
        compute_type: str = "int8",
    ) -> None:
        """
        Initializes the model.
        """
        self.model = WhisperModel(
            checkpoint,
            device=self.device,
            compute_type=compute_type,
        )

    def _inference(self, data: SpeechToTextInput) -> dict:
        """
        Model Inference
        :param      data:           Model Input
        :type       data:           SpeechToTextInput
        :returns:   Model output
        :rtype:     dict
        """
        # Treat strings as base64 encoded numpy array
        audio = (
            np.frombuffer(base64.b64decode(data.query), dtype=np.float32)
            if isinstance(data.query, str)
            else data.query
        )

        # make inference
        segments, _ = self.model.transcribe(
            audio=audio, max_new_tokens=data.max_new_tokens, vad_filter=data.vad_filter
        )
        return {"output": " ".join([s.text.strip() for s in segments])}
