from typing import Optional, AsyncGenerator

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, TextIteratorStreamer

from PIL.Image import Image

from roboml.interfaces import VLLMInput
from roboml.utils import get_quantization_config, pre_process_images_to_pil

from .llm import TransformersLLM


class TransformersMLLM(TransformersLLM):
    """
    Transformers model for VQA.
    """

    def __init__(self, **kwargs):
        """__init__.
        :param kwargs:
        """
        super().__init__(**kwargs)
        # init chat prompt
        self.init_chat_prompt = None

    def _initialize(
        self,
        checkpoint: str = "HuggingFaceM4/idefics2-8b",
        quantization: Optional[str] = "4bit",
        system_prompt: Optional[str] = "You are a helpful AI assistant.",
    ) -> None:
        """Initialize Model.

        :param checkpoint:
        :type checkpoint: str
        :param quantization:
        :type quantization: Optional[str]
        :param history_reset_phrase:
        :type history_reset_phrase: str
        :rtype: None
        """
        self.init_chat_prompt = system_prompt
        quantization_config = get_quantization_config(quantization, self.logger)
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            quantization_config=quantization_config,
            low_cpu_mem_usage=(True if quantization_config else False),
            torch_dtype=torch.float16,
        )
        if not quantization_config:
            self.model.to(self.device)
        self.pre_processor = AutoProcessor.from_pretrained(checkpoint)

    def _inference(self, data: VLLMInput) -> dict | AsyncGenerator:
        """Model inference.
        :param data:
        :param type: VLLMInput
        """
        pil_images: list[Image] = pre_process_images_to_pil(data.images)
        # create prompt
        prompt = self.__create_prompt(data.query, len(data.images))

        text = self.pre_processor.apply_chat_template(
            prompt, add_generation_prompt=True
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
                pil_images,
            )
            return self.consume_streamer(streamer)

        input_ids, generated_ids = self.generate_text(
            text, data.max_new_tokens, data.temperature, None, pil_images
        )

        return self.decode_output(input_ids, generated_ids)

    def __create_prompt(self, query: list[dict], num_images: int) -> list:
        """
        Creates a prompt specific to the model.
        :returns:   Engineered Prompt
        :rtype:     list
        """
        prompt = []
        if self.init_chat_prompt:
            prompt.append({
                "role": "system",
                "content": [{"type": "text", "text": self.init_chat_prompt}],
            })

        # Create hugging face specfic template for Vision2Seq models
        for q in query[:-1]:
            q["content"] = [{"type": "text", "text": q["content"]}]

        # Add image tags to last message
        image_tags = [{"type": "image"} for _ in range(num_images)]
        last_query = image_tags + [{"type": "text", "text": query[-1]["content"]}]
        query[-1]["content"] = last_query

        prompt += query

        self.logger.debug(f"Input to Model: {prompt}")
        return prompt
