from typing import *
import numpy as np
import torch
from .ef_ppo_trainer import EFPPOTrainer
from ef_ppo.discriminator import Discriminator
from ef_ppo import logger

class EFPPOImitationTrainer(EFPPOTrainer):
    """
    EFPPOTrainer that supports imitation objectives
    """
    def __init__(
        self,
        # MDP settings
        environment_reward_weight: float = 1.0,
        # Reference dataset
        reference_dataset_path="dataset.npz",

        # Settings for the discriminator reward calculation
        standardize_discriminator_output=False,
        discriminator_mean_discounting=0.9999,
        imitation_reward_weight=0.1,

        # Discriminator training settings
        discriminator_optimizer=torch.optim.Adam, # type: ignore
        optimizer_kwargs=dict(lr=1e-4),
        discriminator_batch_size=128,
        discriminator_loss_imiation_weight=0.5,
        discriminator_loss_gradience_penalty_weight=0.0,
        discriminator_training_steps=float("inf"),
        update_frozen_discriminator_every=1,
        discriminator_device="cuda",
    ):
        super().__init__()
        self.environment_reward_weight = environment_reward_weight
        self.reference_dataset = np.load(reference_dataset_path)
        self.reference_length = len(self.reference_dataset["observations"])
        self.discriminator = Discriminator(
            self.reference_dataset,
            [512, 256],
            standardize_output=standardize_discriminator_output,
            exponential_mean_discounting=discriminator_mean_discounting,
            imitation_reward_weight=imitation_reward_weight,
            optimizer=discriminator_optimizer, # type: ignore
            optimizer_kwargs=optimizer_kwargs,
            batch_size=discriminator_batch_size,
            weight_imitation=discriminator_loss_imiation_weight,
            weight_gradient_penalty=discriminator_loss_gradience_penalty_weight,
            gradient_steps=discriminator_training_steps,
            update_frozen_every=update_frozen_discriminator_every,
            device=discriminator_device, # type: ignore
        )
        
    def finish_env_step(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray, 
        actions: np.ndarray,
        info: Dict
    ):
        super().finish_env_step(observations, muscle_states, actions, info)
        discriminator_reward = self.discriminator.cost(
            self.agent.last_observations, # type: ignore
            observations,
        )
        info["reward"] *= self.environment_reward_weight
        info["reward"] += discriminator_reward

    def discriminator_update_condition(self) -> bool:
        if self.agent.replay.index == 0: # type: ignore
            return True
        if self.agent.replay.index * self._num_workers % self.reference_length == 0: # type: ignore
            return True
        return False

    def finish_update(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        super().finish_update(observations, muscle_states, actions, info)
        if self.discriminator_update_condition():
            self.discriminator.update(
                self.agent.replay.get_keys( # type: ignore
                    "observations",
                    "next_observations"
                )
            )
