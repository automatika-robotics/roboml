import re
from typing import Literal, Optional

from transformers import AutoModelForImageTextToText, AutoProcessor

from roboml.interfaces import PlanningInput
from roboml.utils import pre_process_images_to_pil, resolve_checkpoint

from ._base import ModelTemplate

# RoboBrain model families with differing prompt and output formats
FAMILY_ROBOBRAIN20 = "robobrain2.0"
FAMILY_ROBOBRAIN25 = "robobrain2.5"

# transformers config model_type to model family
_MODEL_TYPE_FAMILIES = {
    "qwen2_5_vl": FAMILY_ROBOBRAIN20,
    "qwen3_vl": FAMILY_ROBOBRAIN25,
}


def _detect_family(model_type: str, logger) -> str:
    """Detect the RoboBrain family from the loaded model's config type.

    :param model_type:
    :type model_type: str
    :param logger:
    :rtype: str
    """
    family = _MODEL_TYPE_FAMILIES.get(model_type)
    if family is None:
        logger.warning(
            f"Unknown model type '{model_type}' for the RoboBrain2 wrapper. "
            "Assuming RoboBrain 2.5 prompt and output formats."
        )
        return FAMILY_ROBOBRAIN25
    return family


def _supports_thinking(family: str, checkpoint: str) -> bool:
    """Thinking mode is only supported by RoboBrain 2.0 models of size 7B+.

    :param family:
    :type family: str
    :param checkpoint:
    :type checkpoint: str
    :rtype: bool
    """
    return family == FAMILY_ROBOBRAIN20 and "3B" not in checkpoint


def _build_task_text_v25(text: str, task: str) -> str:
    """Task prompt templates from the official RoboBrain 2.5 inference code.

    :param text:
    :type text: str
    :param task:
    :type task: str
    :rtype: str
    """
    # Affordance is handled as a pointing task in RoboBrain 2.5
    if task in ("pointing", "affordance"):
        return (
            f"{text}. Please provide its 2D coordinates. Your answer should be "
            "formatted as a tuple, i.e. [(x, y)], where the tuple contains the x "
            "and y coordinates of a point satisfying the conditions above."
        )
    if task == "trajectory":
        return (
            "Please predict 3D end-effector-centric waypoints to complete the "
            f'task successfully. The task is "{text}". Your answer should be '
            "formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], "
            "where each tuple contains the x and y coordinates and the depth of "
            "the point."
        )
    if task == "grounding":
        return f"Please provide the bounding box coordinate of the region this sentence describes: {text}."
    return text


def _build_task_text_v20(text: str, task: str) -> str:
    """Task prompt templates from the official RoboBrain 2.0 inference code.

    :param text:
    :type text: str
    :param task:
    :type task: str
    :rtype: str
    """
    if task == "pointing":
        return f"{text}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should indicate the normalized pixel locations of the points in the image."
    if task == "affordance":
        return f'You are a robot using the joint control. The task is "{text}". Please predict a possible affordance area of the end effector.'
    if task == "trajectory":
        return f'You are a robot using the joint control. The task is "{text}". Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point.'
    if task == "grounding":
        return f"Please provide the bounding box coordinate of the region this sentence describes: {text}."
    return text


def _build_task_text(text: str, task: str, family: str) -> str:
    """Apply the task specific prompt template of the given model family.

    :param text:
    :type text: str
    :param task:
    :type task: str
    :param family:
    :type family: str
    :rtype: str
    """
    if family == FAMILY_ROBOBRAIN25:
        return _build_task_text_v25(text, task)
    return _build_task_text_v20(text, task)


def _extract_structured_output(answer_text: str, task: str, family: str) -> str | list:
    """Extract structured output from the answer text based on task and family.

    :param answer_text:
    :type answer_text: str
    :param task:
    :type task: str
    :param family:
    :type family: str
    :rtype: str | list
    """
    try:
        if task == "trajectory":
            if family == FAMILY_ROBOBRAIN25:
                # NOTE: RoboBrain 2.5 predicts 3D waypoints with depth.
                # Depth may be integer or decimal; match whole delimited
                # tuples only, so box-style outputs like [x1, y1, x2, y2]
                # are not misread
                trajectory_pattern = (
                    r"[(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*[)\]]"
                )
                points = re.findall(trajectory_pattern, answer_text)
                return [[(int(x), int(y), float(d)) for x, y, d in points]]
            trajectory_pattern = r"(\d+),\s*(\d+)"
            points = re.findall(trajectory_pattern, answer_text)
            return [[(int(x), int(y)) for x, y in points]]
        if task == "pointing" or (
            task == "affordance" and family == FAMILY_ROBOBRAIN25
        ):
            point_pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
            points = re.findall(point_pattern, answer_text)
            return [(int(x), int(y)) for x, y in points]
        if task in ("affordance", "grounding"):
            box_pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
            boxes = re.findall(box_pattern, answer_text)
            return [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]
        return answer_text
    except Exception:
        return "Error occured while extracting structured output"


# Side of the box synthesized around a RoboBrain 2.5 affordance point, as a
# fraction of the image's shorter side.
_AFFORDANCE_BOX_FRACTION = 0.05


def _clamp(value: int, size: int) -> int:
    """Keep a pixel coordinate inside an image dimension."""
    return min(max(value, 0), size - 1)


def _normalize_output(
    answer: str | list, task: str, family: str, image_size: tuple[int, int] | None
) -> tuple[str | list, dict]:
    """Standardize outputs for both 2.0 and 2.5.

    :param answer: Output of :func:`_extract_structured_output`
    :type answer: str | list
    :param task:
    :type task: str
    :param family:
    :type family: str
    :param image_size: (width, height) of the image the model saw, None
        when unknown (nothing is rescaled then)
    :type image_size: tuple[int, int] | None
    :returns: (normalized output, extra response fields)
    :rtype: tuple[str | list, dict]
    """
    extras: dict = {}
    if (
        family != FAMILY_ROBOBRAIN25
        or image_size is None
        or isinstance(answer, str)
        or not answer
    ):
        return answer, extras
    width, height = image_size

    def pixel(x: int | float, y: int | float) -> tuple[int, int]:
        return (
            _clamp(round(x / 1000.0 * width), width),
            _clamp(round(y / 1000.0 * height), height),
        )

    if task == "trajectory":
        # [[(x, y, d), ...]] -> [[(x, y), ...]] with the depths kept seperate
        extras["depths"] = [[d for _, _, d in trajectory] for trajectory in answer]
        waypoints = [[pixel(x, y) for x, y, _ in trajectory] for trajectory in answer]
        return waypoints, extras
    if task == "pointing":
        return [pixel(x, y) for x, y in answer], extras
    if task == "affordance":
        half = max(1, round(_AFFORDANCE_BOX_FRACTION * min(width, height) / 2))
        boxes = []
        for x, y in answer:
            cx, cy = pixel(x, y)
            boxes.append([
                _clamp(cx - half, width),
                _clamp(cy - half, height),
                _clamp(cx + half, width),
                _clamp(cy + half, height),
            ])
        return boxes, extras
    if task == "grounding":
        return [[*pixel(x1, y1), *pixel(x2, y2)] for x1, y1, x2, y2 in answer], extras
    return answer, extras


class RoboBrain2(ModelTemplate):
    """
        RoboBrain 2.0 / 2.5 by BAAI
        @article{RoboBrain2.0TechnicalReport,
        title={RoboBrain 2.0 Technical Report},
        author={BAAI RoboBrain Team},
        journal={arXiv preprint arXiv:2507.02029},
        year={2025}
    }
        @article{tan2026robobrain25depthsight,
      title={RoboBrain 2.5: Depth in Sight, Time in Mind},
      author={Tan, Huajie and Zhou, Enshen and Li, Zhiyu and Xu, Yijie and Ji, Yuheng and Chen, Xiansheng and Chi, Cheng and Wang, Pengwei and Jia, Huizhu and Ao, Yulong and Cao, Mingyu and Chen, Sixiang and Li, Zhe and Liu, Mengzhen and Wang, Zixiao and Rong, Shanyu and Lyu, Yaoxu and Zhao, Zhongxia and Co, Peterson and Li, Yibo and Han, Yi and Xie, Shaoxuan and Yao, Guocai and Wang, Songjing and Zhang, Leiduo and Yang, Xi and Jiao, Yance and Shi, Donghai and Xie, Kunchang and Nie, Shaokai and Men, Chunlei and Lin, Yonghua and Wang, Zhongyuan and Huang, Tiejun and Zhang, Shanghang},
      journal={arXiv preprint arXiv:2601.14352},
      year={2026}
    }
    """

    def __init__(self, **kwargs):
        """__init__.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.family: str = FAMILY_ROBOBRAIN25
        self.supports_thinking: bool = False

    def _initialize(
        self,
        checkpoint: str = "BAAI/RoboBrain2.5-4B",
        source: Optional[Literal["huggingface", "modelscope"]] = None,
    ) -> None:
        """Initialize Model.

        Supports both RoboBrain 2.0 and RoboBrain 2.5 checkpoints; the family
        is detected from the loaded model config.

        RoboBrain 2.5 predicts coordinates relative to a 0-1000 grid, 3D
        (x, y, depth) waypoints for the trajectory task, a point instead of
        a box for the affordance task, and has no thinking mode. Its outputs
        are normalized to the 2.0 contract before they are returned.

        :param checkpoint:
        :type checkpoint: str
        :param source: Hub to download the checkpoint from. Defaults to the
            ROBOML_SOURCE environment variable or huggingface
        :type source: Optional[Literal["huggingface", "modelscope"]]
        :rtype: None
        """
        resolved_checkpoint = resolve_checkpoint(checkpoint, source, self.logger)
        self.model = AutoModelForImageTextToText.from_pretrained(
            resolved_checkpoint, dtype="auto"
        ).to(self.device)

        self.pre_processor = AutoProcessor.from_pretrained(resolved_checkpoint)

        self.family = _detect_family(self.model.config.model_type, self.logger)
        # Thinking mode is only supported by RoboBrain 2.0 models of 7B+
        self.supports_thinking = _supports_thinking(self.family, checkpoint)
        self.logger.info(f"Loaded {self.family} model from {checkpoint}")
        if not self.supports_thinking:
            self.logger.info(
                "Thinking mode is not supported by this checkpoint: the "
                "enable_thinking inference option will be ignored and "
                "'thinking' in the response will always be empty."
            )

    def _inference(self, data: PlanningInput) -> dict:
        """Model inference.
        :param data:
        :param type: PlanningInput
        """
        # create prompt
        prompt = self.__create_prompt(data.query, data.task, len(data.images))

        text = self.pre_processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        # Only append thinking tags for models that support it (2.0 family, 7B+)
        if self.supports_thinking:
            text = (
                f"{text}<think>"
                if data.enable_thinking
                else f"{text}<think></think><answer>"
            )

        # process images
        images = pre_process_images_to_pil(data.images)
        image_size = images[0].size if data.task != "general" and images else None

        inputs = self.pre_processor(
            text=[text],
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # do inference
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=768, do_sample=True, temperature=data.temperature
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]

        generated_text = self.pre_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse thinking and answer from output
        if self.supports_thinking and "</think>" in generated_text:
            parts = generated_text.split("</think>", 1)
            thinking_text = parts[0].replace("<think>", "").strip()
            answer_text = (
                parts[1].replace("<answer>", "").replace("</answer>", "").strip()
            )
        else:
            thinking_text = ""
            answer_text = generated_text.strip()

        answer = _extract_structured_output(answer_text, data.task, self.family)

        if data.task != "general" and (
            isinstance(answer, str) or not answer or answer == [[]]
        ):
            self.logger.warning(
                f"No structured output could be parsed for task '{data.task}' "
                f"from model answer: {answer_text!r}"
            )

        answer, extras = _normalize_output(answer, data.task, self.family, image_size)
        return {"output": answer, "thinking": thinking_text, **extras}

    def __create_prompt(self, query: list[dict], task: str, num_images: int) -> list:
        """
        Creates a prompt specific to the model.
        :returns:   Engineered Prompt
        :rtype:     list
        """
        # Create hugging face specfic template for Vision2Seq models
        for q in query[:-1]:
            q["content"] = [{"type": "text", "text": q["content"]}]

        # RoboBrain family specific task prompts
        text = _build_task_text(query[-1]["content"], task, self.family)

        # Add image tags to last message
        image_tags = [{"type": "image"} for _ in range(num_images)]
        last_query = image_tags + [{"type": "text", "text": text}]
        query[-1]["content"] = last_query

        self.logger.debug(f"Input to Model: {query}")
        return query
