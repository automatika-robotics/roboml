from typing import Optional, Dict, Any

import torch

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

from _base import ModelTemplate
from roboml.interfaces import VLAInput
from roboml.utils import pre_process_single_image_to_np


class LeRobotPolicy(ModelTemplate):
    """
    LeRobot Policy Wrapper (ACT, Diffusion, etc.) using VLAInput.
    """

    POLICY_MAPPING = {
        "act": ACTPolicy,
        "diffusion": DiffusionPolicy,
        "pi0": PI0Policy,
        "smolvla": SmolVLAPolicy,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_metadata: Optional[LeRobotDatasetMetadata] = None
        self.post_processor: Any = None

    def _initialize(
        self,
        checkpoint: str = "fracapuano/robot_learning_tutorial_act_example_model",
        dataset_id: str = "lerobot/svla_so101_pickplace",
        policy_type: str = "act",
    ) -> None:
        """Initialize Policy Model and Processors."""

        # Load Dataset Metadata for Statistics
        self.logger.info(f"Loading metadata for dataset: {dataset_id}")
        self.dataset_metadata = LeRobotDatasetMetadata(dataset_id)

        # Load Policy
        if policy_type.lower() not in self.POLICY_MAPPING:
            raise ValueError(f"Unknown policy type: {policy_type}")

        policy_cls = self.POLICY_MAPPING[policy_type.lower()]
        self.logger.info(f"Loading {policy_type} policy from: {checkpoint}")

        self.model = policy_cls.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # 3. Create Processors (handles Mean/Std normalization)
        self.pre_processor, self.post_processor = make_pre_post_processors(
            self.model.config, dataset_stats=self.dataset_metadata.stats
        )
        self.logger.info("LeRobot policy initialized successfully.")

    def _prepare_observation(self, data: VLAInput) -> Dict[str, torch.Tensor]:
        """
        Manually constructs the model-ready observation tensor dict.
        Replicates logic from `prepare_observation_for_inference` without external deps.
        """
        observation = {}

        # Process State
        # Convert list/np.array -> Tensor, float32, add Batch Dim (1, D), move to device
        state_tensor = torch.tensor(data.state, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        observation["observation.state"] = state_tensor

        # Process Images
        for key, raw_img in data.images.items():
            # Decode to Numpy (H, W, C)
            img_array = pre_process_single_image_to_np(raw_img)

            # Convert to Tensor
            img_tensor = torch.from_numpy(img_array)

            # Normalize [0, 255] -> [0.0, 1.0]
            img_tensor = img_tensor.type(torch.float32) / 255.0

            # Permute (H, W, C) -> (C, H, W)
            if len(img_tensor.shape) == 3:
                img_tensor = img_tensor.permute(2, 0, 1).contiguous()

            # Add Batch Dimension -> (1, C, H, W)
            img_tensor = img_tensor.unsqueeze(0)

            # Move to Device
            img_tensor = img_tensor.to(self.device)

            # Construct key (e.g., 'top' -> 'observation.images.top')
            # Verify against dataset keys to be robust.
            full_key = f"observation.images.{key}"
            observation[full_key] = img_tensor

        observation["task"] = data.task if data.task else ""
        observation["robot_type"] = data.robot_type if data.robot_type else ""

        return observation

    def _inference(self, data: VLAInput) -> Dict[str, Any]:
        """
        Run the policy inference pipeline.
        """
        # Manual Preprocessing (Raw Data -> PyTorch Tensors)
        obs_dict = self._prepare_observation(data)

        # Statistical Normalization (from training Data)
        normalized_obs = self.pre_processor(obs_dict)

        # Model Inference
        with torch.no_grad():
            raw_action = self.model.select_action(normalized_obs)

        # Post-processing (Denormalize)
        denormalized_action = self.post_processor(raw_action)

        return {"output": denormalized_action.squeeze(0).cpu().tolist()}


def main():
    import numpy as np
    import logging

    logger = logging.getLogger("vla")

    print("--- Starting LeRobot Wrapper Execution ---")

    # 1. Create Dummy Input based on provided Example
    # The example used 'side' and 'up' cameras at 640x480.

    print("Creating dummy inputs for 'side' and 'up' cameras (640x480)...")

    # Create random noise images
    dummy_img_side = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_img_up = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # SO-100 Robot typically has 6 joints (SO-100 Follower)
    # We provide a dummy state vector of 6 zeros.
    dummy_state = [0.0] * 6

    input_data = VLAInput(
        images={"side": dummy_img_side, "up": dummy_img_up}, state=dummy_state
    )

    print(f"Input created. State dim: {len(dummy_state)}")

    # 2. Initialize Wrapper (Real Loading)
    policy_wrapper = LeRobotPolicy(logger=logger)

    checkpoint_id = "fracapuano/robot_learning_tutorial_act_example_model"
    dataset_id = "lerobot/svla_so101_pickplace"

    print(f"Initializing Policy from Hub: {checkpoint_id}...")
    print(f"Using Dataset Metadata: {dataset_id}")

    try:
        policy_wrapper._initialize(
            checkpoint=checkpoint_id, dataset_id=dataset_id, policy_type="act"
        )

        # 3. Run Inference
        print("Running Inference...")
        result = policy_wrapper._inference(input_data)

        # 4. Verification
        print("\n--- Inference Result ---")
        if "output" in result:
            action_data = result["output"]
            print(f"Action Type: {type(action_data)}")

            # If it's a list (as converted by our wrapper)
            if isinstance(action_data, list):
                print(f"Action vector: {action_data}")

            print("Execution SUCCESS: Model produced an action.")
        else:
            print("Execution FAILED: No action returned.")

    except Exception as e:
        print(f"\nExecution CRASHED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
