from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# IO Interfaces
class TextToSpeechInput(BaseModel):
    """
    Input values for text to speech inference
    """

    query: str = Field(title="Query string")
    voice: Optional[str] = Field(title="Voice to use", default=None)
    get_bytes: bool = Field(title="Get raw audio bytes", default=False)


class SpeechToTextInput(BaseModel):
    """
    Input values for speech to text inference
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: Union[str, np.ndarray] = Field(
        title="Audio input as base64 encoded or raw np array", min_length=1
    )
    max_new_tokens: Optional[int] = Field(
        title="Maximum number of new tokens to be generated", default=None
    )


class LLMInput(BaseModel):
    """
    Input values for LLM inference
    """

    query: list[dict] = Field(title="Input to the model", min_length=1)
    max_new_tokens: int = Field(
        title="Maximum number of new tokens to be generated", default=100
    )
    temperature: float = Field(
        title="Temperature with which inference is to be generated", default=0.7
    )
    stream: bool = Field(title="Stream output response", default=False)


class VLLMInput(LLMInput):
    """
    Input values for multi modal LLM inference
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: Union[list[str], list[np.ndarray]] = Field(
        title="List of images as base64 strings or numpy arrays", min_length=1
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
