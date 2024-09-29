from typing import Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.pipelines.audio_utils import ffmpeg_read

from roboml.interfaces import SpeechToTextInput
from roboml.ray import app, ingress_decorator
from roboml.utils import b64_str_to_bytes, get_quantization_config

from ._base import ModelTemplate


@ingress_decorator
class Whisper(ModelTemplate):
    """
    Whisper model for Audio to text.
    """

    @app.post("/initialize")
    def _initialize(
        self,
        checkpoint: str = "openai/whisper-small.en",
        quantization: Optional[str] = "4bit",
    ) -> None:
        """
        Initializes the model.
        """
        # Run on GPU with FP16
        quantization_config = get_quantization_config(quantization, self.logger)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=(True if quantization_config else False),
            use_safetensors=True,
            quantization_config=quantization_config,
        )
        if not quantization_config:
            self.model.to(self.device)
        self.pre_processor = AutoProcessor.from_pretrained(checkpoint)

    @app.post("/inference")
    def _inference(self, data: SpeechToTextInput) -> dict:
        """
        Model Inference
        :param      data:           Model Input
        :type       data:           SpeechToTextInput
        :returns:   Model output
        :rtype:     dict
        """
        if isinstance(data.query, str):
            data.query = b64_str_to_bytes(data.query)

        # pre-process audio bytes with ffmpeg
        query = ffmpeg_read(
            data.query, self.pre_processor.feature_extractor.sampling_rate
        )

        # generate input features
        input_features = self.pre_processor(
            query,
            sampling_rate=self.pre_processor.feature_extractor.sampling_rate,
            return_tensors="pt"
        ).input_features.to(self.device, dtype=torch.float16)

        # make inference
        pred_ids = self.model.generate(input_features)
        transcription = self.pre_processor.batch_decode(
            pred_ids, skip_special_tokens=True, decode_with_timestamps=False
        )

        return {"output": transcription[0]}
