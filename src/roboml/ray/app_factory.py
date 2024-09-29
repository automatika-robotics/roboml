from typing import Optional

from fastapi import HTTPException
from ray import available_resources, serve

from roboml import databases, models
from roboml.ray import app, ingress_decorator
from roboml.utils import logger


@serve.deployment()
@ingress_decorator
class AppFactory:
    """App Factory to create ASGI Apps.
    The AppFactory itself is an app."""

    def __init__(
        self, nodes_per_cpu: Optional[int] = None, nodes_per_gpu: Optional[int] = None
    ):
        """__init__."""
        # register all apps created by the factory
        self.app_dict: dict = {}
        self.nodes_per_cpu: int = nodes_per_cpu if nodes_per_cpu else 2
        self.nodes_per_gpu: int = nodes_per_gpu if nodes_per_gpu else 5

    @app.get("/")
    def about(self) -> dict:
        """Server info call.

        :rtype: dict
        """
        return {
            "title": app.title,
            "apps_served": list(self.app_dict.keys()),
            "available_resources": available_resources(),
        }

    @app.post("/add_node", status_code=201)
    def add_node(self, node_name: str, node_type) -> dict:
        """Add a model or database node.

        :param node_name:
        :type node_name: str
        :param node_type:
        :rtype: dict
        """

        if node_name in self.app_dict:
            logger.error(
                f"Name duplication. An node with name {node_name} already exists."
            )
            raise HTTPException(
                status_code=400,
                detail=f"Name duplication. A node with name {node_name} already exists.",
            )

        if hasattr(models, node_type):
            module = getattr(models, node_type)
        # add exception for VisionModel
        elif node_type == "VisionModel":
            try:
                from roboml.models.vision import VisionModel
            except ModuleNotFoundError as e:
                logger.error(
                    "In order to use vision models, install roboml with `pip install roboml[vision]` and install mmcv and mmdetection as explained here, https://github.com/automatika-robotics/robml"
                )
                raise HTTPException(
                    status_code=500,
                    detail="In order to use VisionModel, install roboml with `pip install roboml[vision]` and install mmcv and mmdetection as explained here, https://github.com/automatika-robotics/robml",
                ) from e
            module = VisionModel
        elif hasattr(databases, node_type):
            module = getattr(databases, node_type)
        else:
            logger.error(f"Requested node class {node_type} does not exist")
            raise HTTPException(
                status_code=400,
                detail=f"{node_type} is not a supported model/vectordb type in roboml library. Please use an available model/vectordb type or use another client.",
            )

        # Create a deployment
        deployment = serve.deployment(
            module,
            ray_actor_options={
                "num_cpus": 1 / self.nodes_per_cpu,
                "num_gpus": 1 / self.nodes_per_gpu,
            },
        )
        # Initialize node
        deployment_app = deployment.bind(name=node_name)
        self.app_dict[node_name] = deployment_app
        serve.run(deployment_app, name=node_name, route_prefix=f"/{node_name}")

        logger.info(f"Registered app for {node_name}")
        return {"node": node_name}

    @app.post("/remove_node", status_code=202)
    def remove_node(self, node_name: str) -> dict:
        """Remove a model or database node.

        :param node_name:
        :type node_name: str
        :rtype: dict
        """

        if node_name not in self.app_dict:
            logger.error(f"Node with name {node_name} does not exist.")
            raise HTTPException(
                status_code=400,
                detail=f"Node with name {node_name} does not exists.",
            )

        # Deinitializing node
        serve.delete(node_name)
        del self.app_dict[node_name]

        logger.info(f"Deregistered app for {node_name}")
        return {"node": node_name}
