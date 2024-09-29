import argparse

from ray import serve

from roboml.ray.app_factory import AppFactory
from roboml.resp_server.server import Server
from roboml.utils import logger


def parse_args_for_ray() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Run models and event handlers")
    parser.add_argument(
        "--host", type=str, help="Specify server host address. Default '127.0.0.1'"
    )
    parser.add_argument("--port", type=int, help="Specify server port. Default 8000")
    parser.add_argument(
        "--nodes_per_cpu",
        type=int,
        help="Specify number of nodes to run per CPU. Default None",
    )
    parser.add_argument(
        "--nodes_per_gpu",
        type=int,
        help="Specify number of nodes to run per GPU. Default None",
    )
    return parser.parse_args()


def ray() -> None:
    """Main entry function for ray"""

    args = parse_args_for_ray()
    host = args.host or "0.0.0.0"
    port = args.port or 8000
    nodes_per_cpu = args.nodes_per_cpu or None
    nodes_per_gpu = args.nodes_per_gpu or None
    app_factory = AppFactory.bind(
        nodes_per_cpu=nodes_per_cpu, nodes_per_gpu=nodes_per_gpu
    )
    serve.start(http_options=serve.HTTPOptions(host=host, port=port))
    serve.run(app_factory, name="app_factory", blocking=True)


def parse_args_for_resp() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Run models and event handlers")
    parser.add_argument(
        "--host", type=str, help="Specify server host address. Default '0.0.0.0'"
    )
    parser.add_argument("--port", type=int, help="Specify server port. Default 6379")
    return parser.parse_args()


def resp() -> None:
    """Main entry function for resp"""

    args = parse_args_for_resp()
    host = args.host or "0.0.0.0"
    port = args.port or 6379
    server = Server(logger)
    server._start_server(host=host, port=port)
