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
class AudioInput(BaseModel):
    """
    Input values for audio inference
    """

    query: Union[str, bytes] = Field(title="Audio input raw bytes", min_length=1)


class ImageInput(BaseModel):
    """
    Input values for image inference
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: Union[list[str], list[np.ndarray]] = Field(
        title="List of images as base64 strings or numpy arrays", min_length=1
    )


class TextInput(BaseModel):
    """
    Input values for text inference
    """

    query: Union[str, list[dict]] = Field(title="Input to the model", min_length=1)


class TextToSpeechInput(TextInput):
    """
    Input values for text to speech inference
    """

    voice: Optional[str] = Field(title="Voice to use", default=None)
    get_bytes: bool = Field(title="Get raw audio bytes", default=False)


class SpeechToTextInput(AudioInput):
    """
    Input values for speech to text inference
    """

    max_new_tokens: int = Field(
        title="Maximum number of new tokens to be generated", default=128
    )


class LLMInput(TextInput):
    """
    Input values for LLM inference
    """

    max_new_tokens: int = Field(
        title="Maximum number of new tokens to be generated", default=100
    )
    temperature: float = Field(
        title="Temperature with which inference is to be generated", default=0.7
    )


class VLLMInput(ImageInput, LLMInput):
    """
    Input values for multi modal LLM inference
    """

    pass


class DetectionInput(ImageInput):
    """Input for Detection models."""

    threshold: float = Field(title="Detection confidence threshold", default=0.5)
    get_dataset_labels: bool = Field(
        title="Get dataset label string names", default=True
    )
    labels_to_track: Optional[list[str]] = Field(
        title="List of labels to track. Only used if tracking is enabled during initialization",
        default=None,
    )


# DB Interfaces
class DBAdd(BaseModel):
    """
    Documents to be added to DB
    """

    collection_name: str = Field(title="DB Collection Name")
    ids: list[str] = Field(title="Document IDs", min_length=1)
    metadatas: list[dict] = Field(title="Document metadatas", min_length=1)
    documents: list[str] = Field(title="Documents", min_length=1)
    distance_func: str = Field(
        title="Distance function for the collection", default="l2"
    )
    reset_collection: bool = Field(
        title="Delete existing collection with the name defined in collection_name and create it again with the data provided",
        default=False,
    )


class DBMetadataQuery(BaseModel):
    """
    For retreiving documents based on metadata
    """

    collection_name: str = Field(title="DB Collection Name")
    metadatas: list[dict] = Field(title="Document metadatas", min_length=1)


class DBQuery(BaseModel):
    """
    For retreiving documents based on query
    """

    collection_name: str = Field(title="DB Collection Name")
    query: str = Field(title="Query string")
    n_results: int = Field(title="Number of results", default=1)
