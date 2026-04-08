from typing import Optional, Union

import numpy as np
import supervision as sv
import torch
from trackers import ByteTrackTracker
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from roboml.interfaces import DetectionInput
from roboml.utils import pre_process_images_to_pil

from ._base import ModelTemplate


class VisionModel(ModelTemplate):
    """
    Object detection models from HuggingFace Transformers.
    """

    def __init__(self, **kwargs):
        """__init__.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.trackers: Optional[list[ByteTrackTracker]] = None

    def _initialize(
        self,
        checkpoint: str = "PekingU/rtdetr_r50vd_coco_o365",
        setup_trackers: bool = False,
        num_trackers: int = 1,
        tracking_distance_threshold: int = 30,
    ) -> None:
        """Initialize detection model.
        :param checkpoint: HuggingFace model ID for object detection
        :type checkpoint: str
        :param setup_trackers: Enable object tracking
        :type setup_trackers: bool
        :param num_trackers: Number of tracker instances
        :type num_trackers: int
        :param tracking_distance_threshold: Distance threshold for tracking
        :type tracking_distance_threshold: int
        :rtype: None
        """
        self.pre_processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForObjectDetection.from_pretrained(checkpoint)
        self.model.to(self.device)

        self.data_classes = getattr(self.model.config, "id2label", None)
        if not self.data_classes:
            self.logger.warning(
                "No label mapping found in model config. Integer labels will be used."
            )

        if setup_trackers:
            self.trackers = [
                ByteTrackTracker(
                    minimum_iou_threshold=tracking_distance_threshold / 100.0,
                )
                for _ in range(num_trackers)
            ]

    def _inference(self, data: DetectionInput) -> dict:
        """Model Inference.

        :param data:
        :type data: DetectionInput
        :rtype: dict
        """
        get_dataset_labels = True if data.labels_to_track else data.get_dataset_labels

        # pre-process images to PIL
        pil_images = pre_process_images_to_pil(data.images)

        # run through processor and model
        inputs = self.pre_processor(images=pil_images, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # compute target sizes for post-processing (height, width per image)
        target_sizes = torch.tensor([[img.height, img.width] for img in pil_images]).to(
            self.device
        )

        # post-process to get boxes in pixel coords
        batch_results = self.pre_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=data.threshold
        )

        results = []

        for idx, detection in enumerate(batch_results):
            scores = detection["scores"].cpu().numpy()
            labels = detection["labels"].cpu().numpy()
            bboxes = detection["boxes"].cpu().numpy()

            # Check if any predictions survived thresholding
            if scores.size == 0:
                results.append({})
                continue

            # get text labels from model config
            if get_dataset_labels and self.data_classes:
                labels_text = np.array([
                    self.data_classes.get(int(label), str(label)) for label in labels
                ])
            else:
                labels_text = labels

            result = {
                "bboxes": bboxes.tolist(),
                "labels": labels_text.tolist(),
                "scores": scores.tolist(),
            }

            if data.labels_to_track and self.trackers:
                tracking_result = self._track(
                    data.labels_to_track,
                    self.trackers[idx],
                    scores,
                    labels_text,
                    bboxes,
                )
                result = result | tracking_result

            results.append(result)

        return {"output": results}

    def _filter(
        self,
        criteria: Union[float, np.ndarray],
        scores: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter numpy arrays given threshold or another numpy array."""
        if isinstance(criteria, float):
            mask = scores >= criteria
        elif isinstance(criteria, np.ndarray):
            mask = np.isin(labels, criteria)
        else:
            return scores, labels, bboxes

        return scores[mask], labels[mask], bboxes[mask]

    def _track(
        self,
        labels_to_track: list,
        tracker: ByteTrackTracker,
        scores: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
    ) -> dict:
        """Object Tracking using ByteTrack via trackers library."""
        result = {}
        scores, labels, bboxes = self._filter(
            np.array(labels_to_track), scores, labels, bboxes
        )
        # If labels being tracked dont exist, return
        if scores.size == 0:
            return result

        # Build label-to-id mapping for class_id
        unique_labels = list(set(labels.tolist()))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        class_ids = np.array([label_to_id[label] for label in labels.tolist()])

        detections = sv.Detections(
            xyxy=bboxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=class_ids,
        )

        tracked = tracker.update(detections)

        if len(tracked) > 0:
            # compute center points from tracked bboxes
            centers = (tracked.xyxy[:, :2] + tracked.xyxy[:, 2:]) / 2.0
            # map class_ids back to label strings
            id_to_label = {v: k for k, v in label_to_id.items()}
            tracked_labels = [
                id_to_label.get(cid, "") for cid in tracked.class_id.tolist()
            ]
            result["tracked_bboxes"] = tracked.xyxy.tolist()
            result["tracked_points"] = centers.tolist()
            result["tracked_labels"] = tracked_labels
            result["ids"] = tracked.tracker_id.tolist()

        return result
