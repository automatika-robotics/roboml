import base64
from typing import Literal, Optional

import numpy as np

from faster_whisper import WhisperModel

from roboml.interfaces import SpeechToTextInput
from roboml.utils import CheckpointSource, get_checkpoint_source, resolve_checkpoint

from ._base import ModelTemplate


def _map_size_alias(checkpoint: str) -> str:
    """Map a faster-whisper size alias (e.g. 'small.en') to its repo ID.

    :param checkpoint:
    :type checkpoint: str
    :rtype: str
    """
    from faster_whisper.utils import _MODELS

    return _MODELS.get(checkpoint, checkpoint)


class Whisper(ModelTemplate):
    """
    Whisper model for Audio to text.
    """

    def _initialize(
        self,
        checkpoint: str = "small.en",
        compute_type: str = "int8",
        source: Optional[Literal["huggingface", "modelscope"]] = None,
    ) -> None:
        """
        Initializes the model.

        :param checkpoint: Model size alias (e.g. 'small.en'), repo ID or local path
        :type checkpoint: str
        :param compute_type:
        :type compute_type: str
        :param source: Hub to download the checkpoint from. Defaults to the
            ROBOML_SOURCE environment variable or huggingface
        :type source: Optional[Literal["huggingface", "modelscope"]]
        :rtype: None
        """
        # For ModelScope, map size aliases to their Systran repo IDs and
        # download explicitly; faster-whisper only knows how to pull from HF hub
        if get_checkpoint_source(source) == CheckpointSource.MODELSCOPE.value:
            checkpoint = resolve_checkpoint(
                _map_size_alias(checkpoint), source, self.logger
            )
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
        # Treat strings as base64 encoded bytes
        query = (
            base64.b64decode(data.query) if isinstance(data.query, str) else data.query
        )

        # Treat bytes as 16-bit encoded audio at 16000Hz sample rate
        audio = (
            np.frombuffer(query, dtype=np.int16).astype(np.float32) / 32768
            if isinstance(query, bytes)
            else query
        )

        # make inference
        segments, _ = self.model.transcribe(
            audio=audio,
            max_new_tokens=data.max_new_tokens,
            initial_prompt=data.initial_prompt,
            condition_on_previous_text=True,
            vad_filter=data.vad_filter,
            word_timestamps=data.word_timestamps,
            language=data.language,
        )

        if data.word_timestamps:
            o = []
            # print(list(segments))
            for segment in segments:
                for word in segment.words:
                    w = word.word
                    t = (word.start, word.end, w)
                    o.append(t)
            return {"output": o}

        return {"output": " ".join([s.text.strip() for s in segments])}
