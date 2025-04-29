from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class NodeInit(BaseModel):
    """NodeInit."""

    node_name: str
    node_type: str


class NodeDeinit(BaseModel):
    """NodeDeinit."""

    node_name: str


# IO Interfaces
class TextToSpeechInput(BaseModel):
    """
    Input values for text to speech inference
    """

    query: str = Field(title="Query string")
    voice: Optional[str] = Field(title="Voice to use", default=None)
    get_bytes: bool = Field(title="Get raw audio bytes", default=False)


class AudioInput(BaseModel):
    """
    Input values for audio inference
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: Union[str, bytes, np.ndarray] = Field(
        title="Audio input raw bytes", min_length=1
    )


class SpeechToTextInput(AudioInput):
    """
    Input values for speech to text inference
    """

    max_new_tokens: Optional[int] = Field(
        title="Maximum number of new tokens to be generated", default=None
    )


class DetectionInput(BaseModel):
    """Input for Detection models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: Union[list[str], list[np.ndarray]] = Field(
        title="List of images as base64 strings or numpy arrays", min_length=1
    )
    threshold: float = Field(title="Detection confidence threshold", default=0.5)
    get_dataset_labels: bool = Field(
        title="Get dataset label string names", default=True
    )
    labels_to_track: Optional[list[str]] = Field(
        title="List of labels to track. Only used if tracking is enabled during initialization",
        default=None,
    )
