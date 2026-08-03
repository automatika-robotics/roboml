"""CI-safe tests: chat completion translation and text-only MLLM inference."""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from roboml.chat import is_chat_compatible, translate_chat_request
from roboml.interfaces import (
    ChatCompletionRequest,
    DetectionInput,
    LLMInput,
    PlanningInput,
    VLLMInput,
)
from roboml.models import TransformersMLLM


def _translate(data_model, **request_kwargs):
    return translate_chat_request(ChatCompletionRequest(**request_kwargs), data_model)


class TestChatCompatibility:
    def test_llm_and_mllm_inputs_are_compatible(self):
        assert is_chat_compatible(LLMInput) is True
        assert is_chat_compatible(VLLMInput) is True

    def test_planning_input_is_excluded(self):
        # PlanningInput subclasses VLLMInput but chat completions cannot
        # express its task field; must gate to a 400, not crash with a 500
        assert is_chat_compatible(PlanningInput) is False

    def test_unrelated_inputs_are_excluded(self):
        assert is_chat_compatible(DetectionInput) is False


class TestChatRequestTranslation:
    def test_text_only_on_mllm_node_returns_llm_input(self):
        result = _translate(VLLMInput, messages=[{"role": "user", "content": "Hello"}])
        assert isinstance(result, LLMInput)
        assert not isinstance(result, VLLMInput)

    def test_image_request_on_mllm_node_returns_vllm_input(self):
        result = _translate(
            VLLMInput,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                        },
                    ],
                }
            ],
        )
        assert isinstance(result, VLLMInput)
        assert result.images == ["aGVsbG8="]

    def test_max_tokens_is_forwarded(self):
        result = _translate(
            LLMInput,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=64,
        )
        assert result.max_new_tokens == 64


class TestMLLMTextOnlyInference:
    @pytest.fixture
    def stubbed_mllm(self):
        """TransformersMLLM with mocked model and processor."""
        model = TransformersMLLM(logger=logging.getLogger("test"))
        model.init_chat_prompt = "You are a helpful AI assistant."

        processor_output = MagicMock()
        processor_output.to.return_value = processor_output
        processor_output.input_ids = torch.tensor([[1, 2, 3]])
        model.pre_processor = MagicMock(return_value=processor_output)
        model.pre_processor.apply_chat_template = MagicMock(return_value="TEMPLATED")
        model.pre_processor.decode = MagicMock(return_value="text response")
        model.model = MagicMock()
        model.model.generate.return_value = torch.tensor([[1, 2, 3, 7, 8]])
        return model

    def test_inference_with_plain_llm_input(self, stubbed_mllm):
        """Text-only chat requests reach the MLLM as LLMInput and must work."""
        data = LLMInput(query=[{"role": "user", "content": "Hello"}])
        result = stubbed_mllm._inference(data)
        assert result == {"output": "text response"}

        # no image or video tags in the prompt
        prompt = stubbed_mllm.pre_processor.apply_chat_template.call_args.args[0]
        content_types = [part["type"] for part in prompt[-1]["content"]]
        assert content_types == ["text"]

        # processor called without images or videos
        proc_kwargs = stubbed_mllm.pre_processor.call_args.kwargs
        assert "images" not in proc_kwargs
        assert "videos" not in proc_kwargs
