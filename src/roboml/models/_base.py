from logging import Logger
from abc import abstractmethod
from typing import Any, Optional
from asyncio import AbstractEventLoop
import threading
import time

import torch
from roboml.utils import Status


class ModelTemplate:
    """
    This class describes a model template.
    """

    def __init__(self, logger: Logger):
        self.model: Any = None
        self.pre_processor: Any = None
        self.stream: bool = False
        self.status: Status = Status.LOADED
        self.logger = logger
        self.loop: Optional[AbstractEventLoop] = None
        self.device = self._prepare()

    def _prepare(self):
        _cuda_lock = threading.Lock()
        with _cuda_lock:
            time.sleep(1)  # add a small delay for stability on jetson
            return "cuda" if torch.cuda.is_available() else "cpu"

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
