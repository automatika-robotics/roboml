"""Translation between the OpenAI chat completions format and internal
model inputs."""

from pydantic import BaseModel

from roboml.interfaces import (
    ChatCompletionRequest,
    LLMInput,
    PlanningInput,
    VLLMInput,
)


def is_chat_compatible(data_model: type[BaseModel]) -> bool:
    """Whether a node input model can serve OpenAI chat completions.

    Planning models are excluded due to being task specific.

    :param data_model: Input model of the node handling the request
    :type data_model: type[BaseModel]
    :rtype: bool
    """
    return issubclass(data_model, (LLMInput, VLLMInput)) and not issubclass(
        data_model, PlanningInput
    )


def translate_chat_request(
    request: ChatCompletionRequest, data_model: type[BaseModel]
) -> BaseModel:
    """Translate an OpenAI ChatCompletionRequest to internal LLMInput or VLLMInput.

    :param request:
    :type request: ChatCompletionRequest
    :param data_model: Input model of the node handling the request
    :type data_model: type[BaseModel]
    :rtype: BaseModel
    """
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

    if images and issubclass(data_model, VLLMInput):
        kwargs["images"] = images
        return VLLMInput(**kwargs)

    return LLMInput(**kwargs)
