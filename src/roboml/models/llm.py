from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from roboml.interfaces import LLMInput
from roboml.ray import app, ingress_decorator
from roboml.utils import get_quantization_config

from ._base import ModelTemplate


@ingress_decorator
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
        self.init_chat_prompt = None

    @app.post("/initialize")
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

    @app.post("/inference")
    def _inference(self, data: LLMInput) -> dict:
        """Model inference.
        :param data:
        :param type: LLMInput
        """
        if self.init_chat_prompt:
            data.query.insert(0, {"role": "system", "content": self.init_chat_prompt})

        text = self.pre_processor.apply_chat_template(
            data.query, tokenize=False, add_generation_prompt=True
        )

        inputs = self.pre_processor([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=data.temperature,
                max_new_tokens=data.max_new_tokens,
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                inputs.input_ids, generated_ids, strict=True
            )
        ]

        generated_text = self.pre_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return {"output": generated_text}
