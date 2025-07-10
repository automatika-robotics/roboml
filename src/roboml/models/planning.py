import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from roboml.interfaces import PlanningInput
from roboml.utils import pre_process_images_to_pil

from ._base import ModelTemplate


class RoboBrain2(ModelTemplate):
    """
        RoboBrain2.0 by BAAI
        @article{RoboBrain2.0TechnicalReport,
        title={RoboBrain 2.0 Technical Report},
        author={BAAI RoboBrain Team},
        journal={arXiv preprint arXiv:2507.02029},
        year={2025}
    }
    """

    def __init__(self, **kwargs):
        """__init__.
        :param kwargs:
        """
        super().__init__(**kwargs)

    def _initialize(
        self,
        checkpoint: str = "BAAI/RoboBrain2.0-7B",
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
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint, torch_dtype="auto"
        ).to(self.device)

        self.pre_processor = AutoProcessor.from_pretrained(checkpoint)

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

        # Enable thinking if required
        text = (
            f"{text}<think>"
            if data.enable_thinking
            else f"{text}<think></think><answer>"
        )

        # process images
        images = pre_process_images_to_pil(data.images)

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
        )

        if data.enable_thinking:
            thinking_text = (
                generated_text[0].split("</think>")[0].replace("<think>", "").strip()
            )
            answer_text = (
                generated_text[0]
                .split("</think>")[1]
                .replace("<answer>", "")
                .replace("</answer>", "")
                .strip()
            )
        else:
            thinking_text = ""
            answer_text = (
                generated_text[0]
                .replace("<answer>", "")
                .replace("</answer>", "")
                .strip()
            )

        answer = self._extract_output(answer_text, data.task)

        return {"output": answer, "thinking": thinking_text}

    def __create_prompt(self, query: list[dict], task: str, num_images: int) -> list:
        """
        Creates a prompt specific to the model.
        :returns:   Engineered Prompt
        :rtype:     list
        """
        # Create hugging face specfic template for Vision2Seq models
        for q in query[:-1]:
            q["content"] = [{"type": "text", "text": q["content"]}]

        # Robobrain2.0 specific prompts
        text = query[-1]["content"]  # Content of last message
        if task == "pointing":
            text = f"{text}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should indicate the normalized pixel locations of the points in the image."
        elif task == "affordance":
            text = f'You are a robot using the joint control. The task is "{text}". Please predict a possible affordance area of the end effector.'
        elif task == "trajectory":
            text = f'You are a robot using the joint control. The task is "{text}". Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point.'
        elif task == "grounding":
            text = f"Please provide the bounding box coordinate of the region this sentence describes: {text}."

        # Add image tags to last message
        image_tags = [{"type": "image"} for _ in range(num_images)]
        last_query = image_tags + [{"type": "text", "text": text}]
        query[-1]["content"] = last_query

        self.logger.debug(f"Input to Model: {query}")
        return query

    def _extract_output(self, answer_text: str, task: str) -> str | list:
        """Extract output from the model's answer text based on the task."""
        try:
            if task == "trajectory":
                # Extract trajectory points
                trajectory_pattern = r"(\d+),\s*(\d+)"
                trajectory_points = re.findall(trajectory_pattern, answer_text)
                return [[(int(x), int(y)) for x, y in trajectory_points]]
            elif task == "pointing":
                # Extract points
                point_pattern = r"\(\s*(\d+)\s*,\s*(\d+)\s*\)"
                points = re.findall(point_pattern, answer_text)
                return [(int(x), int(y)) for x, y in points]
            elif task == "affordance":
                # Extract bounding boxes
                box_pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
                boxes = re.findall(box_pattern, answer_text)
                return [
                    [int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes
                ]
            elif task == "grounding":
                # Extract bounding boxes
                box_pattern = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
                boxes = re.findall(box_pattern, answer_text)
                return [
                    [int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes
                ]
            else:
                return answer_text
        except Exception:
            return "Error occured while extracting structured output"
