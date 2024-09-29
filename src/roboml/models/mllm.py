import io
from typing import Optional, Union

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    IdeficsForVisionText2Text,
    AutoModelForVision2Seq,
)

from roboml.interfaces import VLLMInput
from roboml.ray import app, ingress_decorator
from roboml.utils import get_quantization_config, pre_process_images_to_pil

from ._base import ModelTemplate


@ingress_decorator
class Idefics(ModelTemplate):
    """
    Idefics model for VQA.
    """

    def __init__(self, **kwargs):
        """__init__.
        :param kwargs:
        """
        super().__init__(**kwargs)
        # init history buffer for chat history
        self.history_buff: io.StringIO = io.StringIO()
        self.reset_phrase = ""

    @app.post("/initialize")
    def _initialize(
        self,
        checkpoint: str = "HuggingFaceM4/idefics-9b-instruct",
        quantization: Optional[str] = "4bit",
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
        quantization_config = get_quantization_config(quantization, self.logger)
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            quantization_config=quantization_config,
            low_cpu_mem_usage=(True if quantization_config else False),
            torch_dtype=torch.float16,
        )
        if not quantization_config:
            self.model.to(self.device)
        self.pre_processor = AutoProcessor.from_pretrained(checkpoint)

    @app.post("/inference")
    def _inference(self, data: VLLMInput) -> dict:
        """Model inference.
        :param data:
        :param type: VLLMInput
        """
        # create prompt
        prompt = self.__create_prompt(data.query, data.images)

        inputs = self.pre_processor(prompts=prompt, return_tensors="pt").to(
            self.device
        )  # single prompt input
        # Generation args - something specific to idefics
        bad_words_ids = self.pre_processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids

        generated_ids = self.model.generate(
            **inputs,
            bad_words_ids=bad_words_ids,
            do_sample=True,
            temperature=data.temperature,
            max_new_tokens=data.max_new_tokens,
        )
        # get the very last part of the answer - specific to idefics
        generated_text = self.pre_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        output = generated_text.split("\nAssistant:")[-1].strip()
        return {"output": output}

    def __create_prompt(
        self, query: str, images: Union[list[np.ndarray], list[str]]
    ) -> list | None:
        """
        Creates a prompt specific to the model.
        :param      data:    Model Input
        :type       data:    VLLMInput
        :returns:   Engineered Prompt
        :rtype:     list
        """
        template = f"User: {query} |image| <end_of_utterance> |\nAssistant:"
        # prompt needs to be a list for idefics
        prompt = template.split("|")
        img_idx = prompt.index("image")
        # convert image ndarrays to PIL images
        prompt[img_idx] = pre_process_images_to_pil(images, concatenate=True)

        self.logger.debug(f"Input to Model: {prompt}")
        return prompt


@ingress_decorator
class TransformersMLLM(ModelTemplate):
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

    @app.post("/initialize")
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

    @app.post("/inference")
    def _inference(self, data: VLLMInput) -> dict:
        """Model inference.
        :param data:
        :param type: VLLMInput
        """
        pil_images = pre_process_images_to_pil(data.images)

        # create prompt
        prompt = self.__create_prompt(data.query, data.images)

        prompt = self.pre_processor.apply_chat_template(
            prompt, add_generation_prompt=True
        )

        inputs = self.pre_processor(
            images=pil_images, text=prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=data.temperature,
                max_new_tokens=data.max_new_tokens,
            )

        generated_text = self.pre_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        output = generated_text.split("\nAssistant:")[-1].strip()
        return {"output": output}

    def __create_prompt(
        self, query: list[dict], images: Union[list[np.ndarray], list[str]]
    ) -> list:
        """
        Creates a prompt specific to the model.
        :param      data:    Model Input
        :type       data:    VLLMInput
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
        image_tags = [{"type": "image"} for _ in range(len(images))]
        last_query = image_tags + [{"type": "text", "text": query[-1]["content"]}]
        query[-1]["content"] = last_query

        prompt += query

        self.logger.debug(f"Input to Model: {prompt}")
        return prompt
