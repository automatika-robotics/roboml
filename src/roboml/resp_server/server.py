import gc
import inspect
import json
from asyncio.streams import StreamReaderProtocol
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from typing import Callable

import msgpack
import msgpack_numpy as m_pack
import uvicorn
from hiredis import Reader
from pydantic import ValidationError, validate_call

from roboml import models
from roboml.resp_server.stream import StreamReader
from roboml.resp_server.node import RESPNode
from roboml.utils import is_background_task, logging

MAX_CHUNK_SIZE = 65536
OK = b"+OK\r\n"
COMMAND_PING = b"PING"
PONG = b"+PONG\r\n"
COMMAND_QUIT = b"QUIT"
BUILT_IN_COMMANDS = (COMMAND_PING, COMMAND_QUIT)
ERROR_INVALID_COMMAND = b"-INVALID_COMMAND\r\n"
ERROR_INVALID_MSGPACK = b"-INVALID_ARGS\r\n"

# patch msgpack for numpy arrays
m_pack.patch()


class Server:
    """Uvicorn served experimental server.
    Handles RESP based incoming connections."""

    def __init__(self, logger: Logger):
        """__init__.

        :param model_node: Model Node
        """
        self.available_methods = {}
        # register all remote methods present in the server
        self._register_remote_methods()
        self.node_dict = {}
        self.logger = logger
        # max_workers is by default set to min(32, os.cpu_count() + 4)
        self.pool = ThreadPoolExecutor()

    async def _handle_connection(self, reader, writer) -> None:
        """Connection Handler for the RESP based server.

        :param reader:
        :param writer:
        """
        try:
            while True:
                data = await reader.readobj()
                if not data:
                    break
                incoming_command = data[0]
                if incoming_command == b"CLIENT":
                    # Handle redis client identification messages
                    writer.write(OK)
                    await writer.drain()
                    continue
                elif incoming_command == COMMAND_QUIT:
                    writer.write(OK)
                    await writer.drain()
                    break
                elif incoming_command == COMMAND_PING:
                    writer.write(PONG)
                    await writer.drain()
                    continue
                else:
                    try:
                        # get dictionary of defined methods in model node
                        command = self.available_methods[
                            incoming_command.decode("utf-8")
                        ]
                    except KeyError:
                        writer.write(ERROR_INVALID_COMMAND)
                        await writer.drain()
                        continue

                    # pass the command and arguments to execution function
                    self.pool.submit(
                        self._execute, command[0], data[1:], writer, command[1]
                    )
                    await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    def _register_remote_methods(self):
        """Registers all remote methods present in the server"""
        methods = {
            f"{i[0]}": [i[1], is_background_task(i[1])]
            for i in inspect.getmembers(self, predicate=inspect.ismethod)
            if not i[0].startswith("_")
        }
        self.available_methods.update(methods)

    @validate_call
    def add_node(self, node_name: str, node_model: str) -> list[str] | str:
        """Initiate the model node and get all defined methods available"""

        # Check for name duplication
        if node_name in self.node_dict:
            self.logger.warning(
                f"Name duplication. A model/db with name {node_name} already exists."
            )
            return f"Warning: Name duplication. A model/db with name {node_name} already exists."
        if hasattr(models, node_model):
            module = getattr(models, node_model)
        # Add exception for VisionModel
        elif node_model == "VisionModel":
            try:
                from roboml.models.vision import VisionModel
            except ModuleNotFoundError as e:
                self.logger.error(
                    "In order to use vision models, install roboml with `pip install roboml[vision]` and install mmcv and mmdetection as explained here, https://github.com/automatika-robotics/robml"
                )
                raise ModuleNotFoundError(
                    "In order to use VisionModel, install roboml with `pip install roboml[vision]` and install mmcv and mmdetection as explained here, https://github.com/automatika-robotics/robml"
                ) from e
            module = VisionModel
        else:
            self.logger.error(f"Requested node class {node_model} does not exist")
            raise ModuleNotFoundError(
                f"{node_model} is not a supported model type in roboml library. Please use an available model type or use another client."
            )

        # Initialize node
        node = RESPNode(
            name=node_name, model=module(logger=logging.getLogger(node_name))
        )
        self.node_dict[node.name] = node

        # filter out dunder and internal methods and add model name;
        # additionally check method decorators for background_task marking
        # model_name.method_name: [func_signature, is_background_task: bool]
        methods = {
            f"{node.name}.{i[0]}": [i[1], is_background_task(i[1])]
            for i in inspect.getmembers(node, predicate=inspect.ismethod)
            if not i[0].startswith("_")
        }
        self.logger.info(f"Registered methods for {node}")
        self.available_methods.update(methods)
        return list(self.available_methods.keys())

    @validate_call
    def remove_node(self, node_name: str) -> list[str]:
        """Initiate the model node and get all defined methods available"""

        # Check for name duplication
        if node_name not in self.node_dict:
            self.logger.error(f"Node with name {node_name} does not exist.")
            raise ValueError(f"Node with name {node_name} does not exist.")

        # Deinitialize node
        node = self.node_dict[node_name]
        del node
        del self.node_dict[node_name]
        gc.collect()

        # remove available methods starting with node name
        self.logger.info(f"Deregistering methods for {node_name}")
        # get keys in a list as cant iterate directly over the dict at runtime
        for key in list(self.available_methods.keys()):
            if key.split(".")[0] == node_name:
                del self.available_methods[key]
        return list(self.available_methods.keys())

    def _get_args(self, writer, incoming_args) -> dict | None:
        """Unpack args for function input.

        :param command:
        :param incoming_args:
        """
        # unpack args
        try:
            raw_args = msgpack.unpackb(incoming_args)
        except msgpack.UnpackException:
            writer.write(ERROR_INVALID_MSGPACK)
            return None

        return raw_args

    def _execute(
        self,
        command: Callable,
        incoming_args: bytes,
        writer,
        background_task: bool = False,
    ) -> None:
        """Execute remote methods defined in model node.

        :param command: Method Name
        :param incoming_args:
        :param writer:
        """
        try:
            if incoming_args:
                # first argument should be a msgpack msg
                raw_args = self._get_args(writer, incoming_args[0])
                if not raw_args:
                    # write happens in exception handler of the _get_args function
                    return None
            else:
                raw_args = {}

            # if background task then run in threadpool
            if background_task:
                # TODO: Retrieve exceptions here
                self.pool.submit(command, **raw_args)
                writer.write(OK)
                return None
            else:
                result = command(**raw_args)

                # for methods that dont have a return value, return OK
                if not result:
                    writer.write(OK)
                    return None
                resp = msgpack.packb(result)
                writer.write(b"$%d\r\n%b\r\n" % (len(resp), resp))
        except ValidationError as e:
            errors = []
            for idx, err in enumerate(e.errors()):
                loc = json.dumps(err["loc"])
                msg = json.dumps(err["msg"])
                t = json.dumps(err["type"])
                errors.append(f"{idx}: {t} {loc} {msg}")
            writer.write(f"-VALIDATION_ERROR {';'.join(errors)}\r\n".encode("utf8"))
        # raise redis python client compatible ModuleError
        except ModuleNotFoundError:
            writer.write(
                "-ERR Error loading the extension. "
                "Please check the server logs.\r\n".encode("utf8")
            )
        except Exception as e:
            msg = json.dumps(str(e))
            writer.write(f"-UNEXPECTED_ERROR {msg}\r\n".encode("utf8"))

    def _start_server(self, host: str = "127.0.0.1", port: int = 6379, **kwargs):
        # define protocol factory
        # TODO: Use server state and app state for better management
        def factory(config, server_state, app_state, _loop=None):
            self.logger.debug(f"Received {config}:{server_state}:{app_state}")
            reader = StreamReader(limit=MAX_CHUNK_SIZE, loop=_loop)
            reader.set_parser(Reader())
            return StreamReaderProtocol(reader, self._handle_connection, loop=_loop)

        # Set server config
        config = uvicorn.Config(
            self,
            lifespan="off",
            host=host,
            port=port,
            proxy_headers=False,
            http=factory,
            **kwargs,
        )

        # Create server
        server = uvicorn.Server(config=config)

        # run server
        # TODO: Implement multiple workers
        # if config.workers > 1:
        #     sock = config.bind_socket()
        #     supervisor = Multiprocess(config, target=server.run, sockets=[sock])
        #     supervisor.run()
        # else:
        server.run()
        self.pool.shutdown()
