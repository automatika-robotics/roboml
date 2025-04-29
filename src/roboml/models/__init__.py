from ._base import ModelTemplate
from .speech_to_text import Whisper
from .text_to_speech import SpeechT5

__all__ = [
    "ModelTemplate",
    "Whisper",
    "SpeechT5",
]
