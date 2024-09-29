import numpy as np
from pydantic import validate_call
from sentence_transformers import SentenceTransformer

from ._base import ModelTemplate


class EncodingModel(ModelTemplate):
    """
    Encoding model for text. Used internally by language models
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embed_instruction: str = ""
        self.query_instruction: str = ""
        self.BGE_EMBED_INSTRUCTION_EN: str = "Represent the document for retrieval: "
        self.BGE_EMBED_INSTRUCTION_ZH: str = "表示要检索的文档："
        self.BGE_QUERY_INSTRUCTION_EN: str = (
            "Represent this question for searching relevant passages: "
        )
        self.BGE_QUERY_INSTRUCTION_ZH: str = "为这个句子生成表示以用于检索相关文章："

    @validate_call
    def _initialize(self, checkpoint: str = "BAAI/bge-small-en", **init_kwargs) -> None:
        """
        Initializes the model.
        """
        model_kwargs = {}
        model_kwargs.update(init_kwargs)
        # set device in kwargs
        model_kwargs["device"] = self.device
        self.model = SentenceTransformer(checkpoint, **model_kwargs)
        if "bge" in checkpoint:
            if "-zh" in checkpoint:
                self.embed_instruction = self.BGE_EMBED_INSTRUCTION_ZH
                self.query_instruction = self.BGE_QUERY_INSTRUCTION_ZH
            else:
                self.embed_instruction = self.BGE_EMBED_INSTRUCTION_EN
                self.query_instruction = self.BGE_QUERY_INSTRUCTION_EN
        else:
            self.embed_instruction = ""
            self.query_instruction = ""

    @validate_call
    def _inference(self, *_):
        raise NotImplementedError("Encoder does not have an inference method")

    @validate_call
    def embed_documents(self, data: list[str], **encode_kwargs) -> list[list[float]]:
        """
        Sends back encoded documents given a list of documents
        :param      data:           Model Input
        :type       data:           EncodingInferenceInput
        """
        encode_kwargs["normalize_embeddings"] = False
        docs = [self.embed_instruction + doc.replace("\n", " ") for doc in data]
        embedding = self.model.encode(docs, **encode_kwargs)
        return embedding.tolist()

    @validate_call
    def embed_query(self, data: str, **encode_kwargs) -> np.ndarray:
        """
        Sends back encoded query given a text query
        :param      data:           Model Input
        :type       data:           EncodingInferenceInput
        """
        encode_kwargs["normalize_embeddings"] = False
        query = self.query_instruction + data.replace("\n", " ")
        return self.model.encode(query, **encode_kwargs)
