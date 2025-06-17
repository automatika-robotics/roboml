from typing import Optional, AsyncGenerator
from queue import Empty
import asyncio

import torch
from PIL.Image import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from roboml.interfaces import LLMInput
from roboml.utils import get_quantization_config

from ._base import ModelTemplate


class TransformersLLM(ModelTemplate):
    """
    Transformers LLM.
    """

    def __init__(self, **kwargs):
        """__init__.
        :param kwargs:
        """

        super().__init__(**kwargs)
        # init chat prompt
        self.init_chat_prompt: Optional[str] = None

    def _initialize(
        self,
        checkpoint: str = "microsoft/Phi-3-mini-4k-instruct",
        quantization: Optional[str] = "4bit",
        system_prompt: Optional[str] = "You are a helpful AI assistant.",
    ) -> None:
        """Initialize Model.

        :param checkpoint:
        :type checkpoint: str
        :param quantization:
        :type quantization: Optional[str]
        :param init_chat_prompt:
        :type init_chat_prompt: str
        :rtype: None
        """
        self.init_chat_prompt = system_prompt
        quantization_config = get_quantization_config(quantization, self.logger)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=quantization_config,
            low_cpu_mem_usage=(True if quantization_config else False),
            torch_dtype=torch.float16,
        )
        if not quantization_config:
            self.model.to(self.device)
        self.pre_processor = AutoTokenizer.from_pretrained(checkpoint)

    def _inference(self, data: LLMInput) -> dict | AsyncGenerator:
        """Model inference.
        :param data:
        :param type: LLMInput
        """
        if self.init_chat_prompt:
            data.query.insert(0, {"role": "system", "content": self.init_chat_prompt})

        text = self.pre_processor.apply_chat_template(
            data.query, tokenize=False, add_generation_prompt=True
        )

        if data.stream:
            streamer = TextIteratorStreamer(
                self.pre_processor,
                timeout=0,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            self.loop.run_in_executor(
                None,
                self.generate_text,
                text,
                data.max_new_tokens,
                data.temperature,
                streamer,
            )
            return self.consume_streamer(streamer)

        input_ids, generated_ids = self.generate_text(
            text, data.max_new_tokens, data.temperature, None
        )

        return self.decode_output(input_ids, generated_ids)

    def generate_text(
        self,
        text: str,
        max_new_tokens: int,
        temperature: float,
        streamer: Optional[TextIteratorStreamer],
        images: Optional[list[Image]] = None,
    ):
        input = (
            self.pre_processor(text=text, return_tensors="pt")
            if not images
            else self.pre_processor(text=text, images=images, return_tensors="pt")
        ).to(self.device)
        # input_ids = input.input_ids.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **input,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )
        if not streamer:
            return input.input_ids, generated_ids

    def decode_output(self, input_ids, generated_ids):
        # Remove prompt tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(input_ids, generated_ids, strict=True)
        ]

        # Decode to get text
        generated_text = self.pre_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return {"output": generated_text}

    async def consume_streamer(self, streamer: TextIteratorStreamer):
        while True:
            try:
                for token in streamer:
                    yield token
                break
            except Empty:
                await asyncio.sleep(0.001)
