from logging import Logger

from pydantic import ValidationError, validate_call
import inspect

from . import app, ingress_decorator
from roboml.models import ModelTemplate
from roboml.utils import Status, logging


@ingress_decorator
class RayHTTPNode:
    """
    This class describes a node that gets deployed as a ray app.
    """

    def __init__(
        self, *, name: str, model: ModelTemplate, init_timeout: int = 600, **_
    ):
        self.name: str = name
        self.model: ModelTemplate = model
        self.init_timeout: int | None = init_timeout  # 10 minutes
        self.logger: Logger = logging.getLogger(self.name)

    @app.post("/initialize")
    def initialize(self, body: dict | None = None) -> None:
        """
        Initialize the model associated with this node
        """
        if self.model.status == Status.READY:
            return

        self.model.status = Status.INITIALIZING
        try:
            if body:
                validated_init = validate_call(self.model._initialize)
                validated_init(**body)
            else:
                self.model._initialize()
        except Exception as e:
            self.logger.error(f"Initialization Error: {e}")
            self.model.status = Status.INITIALIZATION_ERROR
            raise e
        self.logger.info(f"{self.__class__.__name__} Model initialized")
        self.model.status = Status.READY

    @app.post("/inference")
    def inference(self, body: dict) -> dict:
        """
        Calls the models inference function and sets status accordingly
        """
        if self.model.status is not Status.READY:
            self.logger.error("Error: Inference called before initialization")
            raise Exception("Inference called before initialization")
        try:
            data_model = (
                inspect.signature(self.model._inference).parameters["data"].annotation
            )
            data = data_model(**body)
        except ValidationError:
            self.logger.error("Validation Error occured for inference input")
            raise
        result = self.model._inference(data)
        return result

    def get_status(self):
        """Returns status of the model node"""
        return self.model.status.name
