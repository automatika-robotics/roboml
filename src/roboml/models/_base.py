from logging import Logger
from abc import abstractmethod
from typing import Any, Optional
from asyncio import AbstractEventLoop

import torch
from roboml.utils import Status


class ModelTemplate:
    """
    This class describes a model template.
    """

    def __init__(self, logger: Logger):
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Any = None
        self.pre_processor: Any = None
        self.stream: bool = False
        self.status: Status = Status.LOADED
        self.logger = logger
        self.loop: Optional[AbstractEventLoop] = None

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
    def _inference(self, *_, **__) -> Any:
        """Model specific inference function, to be implemented in derived classes.
        :rtype: dict
        """
        raise NotImplementedError(
            "Model specific inference needs to be implemented by derived model classes"
        )
