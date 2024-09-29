from abc import abstractmethod
from logging import Logger
import inspect

from pydantic import validate_call, ValidationError


from roboml.models._encoding import EncodingModel
from roboml.utils import Status, logging


class VectorDBTemplate:
    """
    This class describes a VectorDB template.
    """

    def __init__(self, *, name: str, **_):
        """__init__.
        :param name:
        :type name: str
        :param _:
        """
        self.name: str = name
        self.encoding_model: EncodingModel
        self.logger: Logger = logging.getLogger(self.name)
        self.status: Status = Status.LOADED

    def initialize(self, **kwargs) -> None:
        """
        Calls the db initialization function and sets status accordingly
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
        self.logger.info(f"{self.__class__.__name__} VectorDB initialized")
        self.status = Status.READY

    def get_status(self):
        """Returns status of the model node"""
        return self.status.name

    def add(self, **kwargs) -> dict:
        """VectorDB specific add function.
        :param kwargs:
        :rtype: None
        """
        if self.status is not Status.READY:
            self.logger.error("Error: DB method called before initialization")
            raise Exception("DB Method Called called before initialization")
        try:
            data_model = inspect.signature(self._add).parameters["data"].annotation
            data = data_model(**kwargs)
        except ValidationError:
            self.logger.error("Validation Error occured for inference input")
            raise
        return self._add(data)

    def conditional_add(self, **kwargs) -> dict:
        """VectorDB specific conditional_add function.
        :param kwargs:
        :rtype: None
        """
        if self.status is not Status.READY:
            self.logger.error("Error: DB method called before initialization")
            raise Exception("DB Method Called called before initialization")
        try:
            data_model = (
                inspect.signature(self._conditional_add).parameters["data"].annotation
            )
            data = data_model(**kwargs)
        except ValidationError:
            self.logger.error("Validation Error occured for inference input")
            raise
        return self._conditional_add(data)

    def metadata_query(self, **kwargs) -> dict:
        """VectorDB specific metadata_query function.
        :param kwargs:
        :rtype: None
        """
        if self.status is not Status.READY:
            self.logger.error("Error: DB method called before initialization")
            raise Exception("DB Method Called called before initialization")
        try:
            data_model = (
                inspect.signature(self._metadata_query).parameters["data"].annotation
            )
            data = data_model(**kwargs)
        except ValidationError:
            self.logger.error("Validation Error occured for inference input")
            raise
        return self._metadata_query(data)

    def query(self, **kwargs) -> dict:
        """VectorDB specific query function.
        :param kwargs:
        :rtype: None
        """
        if self.status is not Status.READY:
            self.logger.error("Error: DB method called before initialization")
            raise Exception("DB Method Called called before initialization")
        try:
            data_model = inspect.signature(self._query).parameters["data"].annotation
            data = data_model(**kwargs)
        except ValidationError:
            self.logger.error("Validation Error occured for inference input")
            raise
        return self._query(data)

    @abstractmethod
    def _initialize(self, *_, **__) -> None:
        """VectorDB specific initialize function, to be implemented in derived classes.
        :param kwargs:
        :rtype: None
        """
        raise NotImplementedError(
            "VectorDB specific initialize method needs to be implemented by derived classes"
        )

    @abstractmethod
    def _add(self, *_, **__) -> dict:
        """VectorDB specific add function, to be implemented in derived classes.
        :param kwargs:
        :rtype: None
        """
        raise NotImplementedError(
            "VectorDB specific add method needs to be implemented by derived classes"
        )

    @abstractmethod
    def _conditional_add(self, *_, **__) -> dict:
        """VectorDB specific conditional_add function, to be implemented in derived classes.
        :param kwargs:
        :rtype: None
        """
        raise NotImplementedError(
            "VectorDB specific conditional_add method needs to be implemented by derived classes"
        )

    @abstractmethod
    def _metadata_query(self, *_, **__) -> dict:
        """VectorDB specific metadata_query function, to be implemented in derived classes.
        :param kwargs:
        :rtype: None
        """
        raise NotImplementedError(
            "VectorDB specific metadata_query method needs to be implemented by derived classes"
        )

    @abstractmethod
    def _query(self, *_, **__) -> dict:
        """VectorDB specific query function, to be implemented in derived classes.
        :param kwargs:
        :rtype: None
        """
        raise NotImplementedError(
            "VectorDB specific query method needs to be implemented by derived classes"
        )
