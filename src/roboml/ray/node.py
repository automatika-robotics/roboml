from logging import Logger
import asyncio
from typing import AsyncGenerator

import msgpack_numpy as m_pack
import msgpack
from pydantic import BaseModel, ValidationError, validate_call
from fastapi import (
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
)
from starlette.responses import StreamingResponse
import inspect

from . import app, ingress_decorator
from roboml.models import ModelTemplate
from roboml.interfaces import (
    LLMInput,
    VLLMInput,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)
from roboml.utils import Status, logging

# patch msgpack for numpy
m_pack.patch()

# Define a constant for the chunk size (1MB)
CHUNK_SIZE = 1 * 1024 * 1024

# Model types that support chat completions
CHAT_COMPATIBLE_INPUTS = (LLMInput, VLLMInput)


@ingress_decorator
class RayNode:
    """
    This class describes a node that gets deployed as a ray app.
    """

    def __init__(
        self, *, name: str, model: type[ModelTemplate], init_timeout: int = 600, **_
    ):
        self.name: str = name
        self.init_timeout: int | None = init_timeout  # 10 minutes
        self.logger: Logger = logging.getLogger(self.name)
        self.model: ModelTemplate = model(logger=self.logger)
        self.model.loop = asyncio.get_running_loop()
        self.data_model: type[BaseModel] = BaseModel

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

            # Get data model for checking during inference
            self.data_model = (
                inspect.signature(self.model._inference).parameters["data"].annotation
            )

        except Exception as e:
            self.logger.error(f"Initialization Error: {e}")
            self.model.status = Status.INITIALIZATION_ERROR
            raise HTTPException(
                status_code=500, detail=f"Initialization Error: {e}"
            ) from e
        self.logger.info(f"{self.__class__.__name__} Model initialized")
        self.model.status = Status.READY

    @app.post("/inference")
    def inference(self, body: dict) -> Response:
        """
        Calls the models inference function
        """
        if self.model.status is not Status.READY:
            self.logger.error("Inference called before initialization")
            raise HTTPException(
                status_code=500, detail="Inference called before initialization"
            )
        # verify model input
        try:
            data = self.data_model(**body)
            result = self.model._inference(data)
        except ValidationError as e:
            self.logger.error("Validation Error occured for inference input")
            raise HTTPException(status_code=400, detail=e.errors()) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        # Send streaming response if set in the model
        if isinstance(result, AsyncGenerator):
            return StreamingResponse(result, media_type="text/plain")
        return result

    @app.post("/v1/chat/completions")
    async def chat_completions(self, body: dict) -> Response:
        """
        OpenAI-compatible chat completions endpoint.
        Translates between OpenAI format and internal LLMInput/VLLMInput.
        """
        if self.model.status is not Status.READY:
            raise HTTPException(status_code=500, detail="Model not initialized")

        if not issubclass(self.data_model, CHAT_COMPATIBLE_INPUTS):
            raise HTTPException(
                status_code=400,
                detail="This model does not support chat completions. "
                "Use /inference instead.",
            )

        try:
            request = ChatCompletionRequest(**body)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors()) from e

        # Translate OpenAI format to internal format
        try:
            internal_input = self._translate_chat_request(request)
            result = self.model._inference(internal_input)
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=e.errors()) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        # Handle streaming
        if isinstance(result, AsyncGenerator):
            return StreamingResponse(
                self._stream_chat_response(result),
                media_type="text/event-stream",
            )

        # Wrap in OpenAI response format
        text = result.get("output", "")
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=str(text)),
                )
            ]
        )
        return Response(
            content=response.model_dump_json(),
            media_type="application/json",
        )

    def _translate_chat_request(self, request: ChatCompletionRequest) -> BaseModel:
        """Translate OpenAI ChatCompletionRequest to internal LLMInput or VLLMInput."""
        images = []
        query = []

        for msg in request.messages:
            content = msg.get("content", "")

            # Handle multimodal content (list of content parts)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        # Strip data URI prefix if present
                        if url.startswith("data:"):
                            url = url.split(",", 1)[-1]
                        images.append(url)
                query.append({
                    "role": msg.get("role", "user"),
                    "content": " ".join(text_parts),
                })
            else:
                query.append({"role": msg.get("role", "user"), "content": content})

        kwargs = {
            "query": query,
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }

        if images and issubclass(self.data_model, VLLMInput):
            kwargs["images"] = images
            return VLLMInput(**kwargs)

        return LLMInput(**kwargs)

    async def _stream_chat_response(self, generator: AsyncGenerator):
        """Wrap streaming tokens in OpenAI SSE format."""
        from uuid import uuid4

        stream_id = f"chatcmpl-{uuid4().hex[:12]}"

        # First chunk with role
        chunk = ChatCompletionChunk(
            id=stream_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(role="assistant"),
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        # Content chunks
        async for token in generator:
            chunk = ChatCompletionChunk(
                id=stream_id,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(content=token),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        chunk = ChatCompletionChunk(
            id=stream_id,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    async def _send_chunked(self, ws: WebSocket, data: str | bytes) -> None:
        """
        Sends data over the websocket, chunking it if it's larger than CHUNK_SIZE.
        - For bytes, data is split into chunks of CHUNK_SIZE bytes.
        - For strings, data is encoded to check byte size, but sliced by character
          count to preserve valid text frames to avoid splitting multi-byte characters.
        """
        if isinstance(data, str):
            # Check the encoded byte size of the string
            if len(data.encode("utf-8")) > CHUNK_SIZE:
                for i in range(0, len(data), CHUNK_SIZE):
                    await ws.send_text(data[i : i + CHUNK_SIZE])
            else:
                await ws.send_text(data)

        elif isinstance(data, bytes):
            if len(data) > CHUNK_SIZE:
                for i in range(0, len(data), CHUNK_SIZE):
                    await ws.send_bytes(data[i : i + CHUNK_SIZE])
            else:
                await ws.send_bytes(data)

    @app.websocket("/ws_inference")
    async def handle_request(self, ws: WebSocket) -> None:
        """Websockets endpoint that streams text data for text generation models"""
        await ws.accept()
        try:
            while True:
                raw_data = await ws.receive_bytes()
                data_dict = msgpack.unpackb(raw_data)
                # verify model input
                data = self.data_model(**data_dict)

                # check for termination signal
                if getattr(data, "query", None) == b"\r\n":
                    await ws.send_bytes(msgpack.packb(data.query))
                    continue

                result = self.model._inference(data)

                if isinstance(result, AsyncGenerator):
                    async for res in result:
                        await ws.send_text(res)
                    await ws.send_text("<<Response Ended>>")
                else:
                    res = result["output"]
                    if isinstance(res, (str, bytes)):
                        await self._send_chunked(ws, res)
                    else:
                        # For other types, pack them first then send with chunking.
                        # TODO: Currently assumes that msgpack data is
                        # always smaller than CHUNK_SIZE
                        await self._send_chunked(ws, msgpack.packb(res))

        except ValidationError as e:
            self.logger.error("Validation Error occured for inference input")
            raise WebSocketException(code=1007, reason=f"{e.errors()}") from e
        except WebSocketDisconnect:
            self.logger.info("Websocket client disconnected.")
        except Exception as e:
            self.logger.error(f"Unknown exception: {e}")
            raise WebSocketException(code=1007, reason=f"{e}") from e

    @app.get("/get_status")
    def get_status(self):
        """Returns status of the model node"""
        return self.model.status.name
