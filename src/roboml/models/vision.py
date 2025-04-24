from typing import Optional, Union

import numpy as np
import torch
from platformdirs import user_cache_dir

from roboml.interfaces import DetectionInput
from ._base import ModelTemplate

from roboml.utils import (
    get_mmdet_model,
    pre_process_images_to_np,
    convert_with_mmdeploy,
)
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose
from norfair import Detection, Tracker


class VisionModel(ModelTemplate):
    """
    Object detection models from MMDetection.
    """

    def __init__(self, **kwargs):
        """__init__.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.trackers: Optional[list[Tracker]] = None

    def _initialize(
        self,
        checkpoint: str = "dino-4scale_r50_8xb2-12e_coco",
        cache_dir: str = "mmdet",
        setup_trackers: bool = False,
        num_trackers: int = 1,
        tracking_distance_function: str = "euclidean",
        tracking_distance_threshold: int = 30,
        deploy_tensorrt: bool = False,
    ) -> None:
        """_initialize.
        :param cache_dir:
        :type cache_dir: str
        :param _:
        :rtype: None
        """
        self.deploy_tensorrt = deploy_tensorrt
        # check if the checkpoint exists in cache else download it
        cache = user_cache_dir(cache_dir)
        config, weights = get_mmdet_model(cache, checkpoint, self.logger)

        self.model = init_detector(config, weights, device=self.device)

        self.data_classes = self.model.dataset_meta["classes"]

        if deploy_tensorrt:
            # deploy tensorrt version if asked
            try:
                import tensorrt
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "NVIDIA tensorrt needs to be installed for TensorRT deployment. Find install instructions on this link https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
                ) from e
            self.logger.info(f"Found tensorrt version {tensorrt.__version__}")
            self.model, self.input_shape, self.task_processor = convert_with_mmdeploy(
                config, weights, self.device, cache
            )
        else:
            # other build test pipeline
            test_pipeline = get_test_pipeline_cfg(self.model.cfg)
            test_pipeline[0].type = "mmdet.LoadImageFromNDArray"

            self.test_pipeline = Compose(test_pipeline)

        if setup_trackers:
            self.trackers = [
                Tracker(
                    distance_function=tracking_distance_function,
                    distance_threshold=tracking_distance_threshold,
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

        # pre-process images if strings received
        images = pre_process_images_to_np(data.images)
        detections = []

        for img in images:
            if self.deploy_tensorrt:
                model_inputs, _ = self.task_processor.create_input(
                    img, self.input_shape
                )
                with torch.no_grad():
                    # result is already a list
                    detections += self.model.test_step(model_inputs)
            else:
                data_ = {"img": img}
                # build the data pipeline
                data_ = self.test_pipeline(data_)

                # mmdet preprocessing qwirks
                data_["inputs"] = [data_["inputs"]]
                data_["data_samples"] = [data_["data_samples"]]

                with torch.no_grad():
                    detections.append(self.model.test_step(data_)[0])

        results = []

        # filter and convert results from DetObject tensors to lists
        for idx, detection in enumerate(detections):
            result = {}
            if "pred_instances" in detection and "bboxes" in detection.pred_instances:
                scores = detection.pred_instances.scores.cpu().numpy()
                labels = detection.pred_instances.labels.cpu().numpy()
                bboxes = detection.pred_instances.bboxes.cpu().numpy()
                scores, labels, bboxes = self._filter(
                    data.threshold, scores, labels, bboxes
                )
                # Check if predictions survived thresholding
                if not (scores.size == 0):
                    # if labels are requested in text
                    if get_dataset_labels:
                        # get text labels from model dataset info
                        labels = np.vectorize(
                            lambda x: self.data_classes[x],
                        )(labels)

                result = {
                    "bboxes": bboxes.tolist(),
                    "labels": labels.tolist(),
                    "scores": scores.tolist(),
                }
                if data.labels_to_track and self.trackers:
                    tracking_result = self._track(
                        data.labels_to_track,
                        self.trackers[idx],
                        scores,
                        labels,
                        bboxes,
                    )
                    result = result | tracking_result

            if result:
                results.append(result)

        return {"output": results}

    def _filter(
        self,
        criteria: Union[float, np.ndarray],
        scores: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter numpy arrays given threshold or another numpy array"""
        if isinstance(criteria, float):
            # filter by score threshold
            mask = scores >= criteria
        elif isinstance(criteria, np.ndarray):
            # filter by labels
            mask = np.isin(labels, criteria)
        else:
            return scores, labels, bboxes

        return scores[mask], labels[mask], bboxes[mask]

    def _track(
        self,
        labels_to_track: list,
        tracker: Tracker,
        scores: np.ndarray,
        labels: np.ndarray,
        bboxes: np.ndarray,
    ) -> dict:
        """Object Tracking using norfair Tracker"""
        result = {}
        scores, labels, bboxes = self._filter(
            np.array(labels_to_track), scores, labels, bboxes
        )
        # If labels being tracked dont exist, return
        if scores.size == 0:
            return result
        detections_to_track = []
        for label, bbox in zip(labels, bboxes, strict=False):
            box = bbox.reshape(2, 2)
            detections_to_track.append(
                Detection(np.vstack([box, box.mean(axis=0)]), label=label)
            )

        tracked_objects = tracker.update(detections=detections_to_track)

        if tracked_objects:
            result["tracked_points"] = []
            result["tracked_labels"] = []
            result["ids"] = []
            result["estimated_velocities"] = []
            for obj in tracked_objects:
                result["tracked_points"].append(obj.estimate.tolist())
                result["tracked_labels"].append(obj.label)
                result["ids"].append(obj.id)
                result["estimated_velocities"].append(obj.estimate_velocity.tolist())

        return result
