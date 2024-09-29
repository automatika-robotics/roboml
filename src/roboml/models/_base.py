from abc import abstractmethod
from logging import Logger
from threading import Lock
from typing import Any, Optional

from pydantic import ValidationError, validate_call
import torch
import inspect

from roboml.utils import Status, logging


class ModelTemplate:
    """
    This class describes a model template.
    """

    def __init__(self, *, name: str, init_timeout: int = 600, **_):
        self.name: str = name
        self.init_timeout: Optional[int] = init_timeout  # 10 minutes
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Any = None
        self.pre_processor: Any = None
        self.status: Status = Status.LOADED
        self.logger: Logger = logging.getLogger(self.name)
        self.lock = Lock()

    def initialize(self, **kwargs) -> None:
        """
        Calls the models initialization function and sets status accordingly
        """
        if self.status == Status.READY:
            return

        self.status = Status.INITIALIZING
        try:
            validated_init = validate_call(self._initialize)
            validated_init(**kwargs)
        except Exception as e:
            self.logger.error(f"Initialization Error: {e}")
            self.status = Status.INITIALIZATION_ERROR
            raise e
        self.logger.info(f"{self.__class__.__name__} Model initialized")
        self.status = Status.READY

    def inference(self, **kwargs) -> dict:
        """
        Calls the models inference function and sets status accordingly
        """
        if self.status is not Status.READY:
            self.logger.error("Error: Inference called before initialization")
            raise Exception("Inference called before initialization")
        try:
            data_model = (
                inspect.signature(self._inference).parameters["data"].annotation
            )
            data = data_model(**kwargs)
        except ValidationError:
            self.logger.error("Validation Error occured for inference input")
            raise
        with self.lock:
            result = self._inference(data)
        return result

    def get_status(self):
        """Returns status of the model node"""
        return self.status.name

    @abstractmethod
    def _initialize(self, *_, **__) -> None:
        """Model specific initialization function, to be implemented in derived classes.
        :param kwargs:
        :rtype: None
        """
        raise NotImplementedError(
            "Model specific initializaion needs to be implemented by derived model classes"
        )

    @abstractmethod
    def _inference(self, *_, **__) -> dict:
        """Model specific inference function, to be implemented in derived classes.
        :rtype: dict
        """
        raise NotImplementedError(
            "Model specific inference needs to be implemented by derived model classes"
        )
