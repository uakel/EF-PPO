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
        environment_constraint_weight: float = 1.0,

        # Reference dataset
        reference_dataset_path: str ="dataset.npz",

        # Settings for the discriminator reward calculation
        reward_shaping: Literal["neg_fancy", "fancy", "vanilla", "none"]="none",
        constraint_shaping: Literal["neg_fancy", "fancy", "vanilla", "none"]="none",
        discriminator_mean_discounting: float=0.9999,
        imitation_reward_weight: float=0.0,
        imitation_constraint_weight: float=1.0,
        imitation_constraint_slack: float=0.2,

        # Discriminator training settings
        discriminator_optimizer: torch.optim.Optimizer=torch.optim.Adam, # type: ignore
        optimizer_kwargs: Dict=dict(lr=0.7e-4),
        discriminator_batch_size: int=32,
        discriminator_loss_imiation_weight: float=0.5,
        discriminator_loss_gradience_penalty_weight: float=0.0,
        discriminator_training_steps: float=float("inf"),
        discriminator_device: Literal["cuda", "cpu"]="cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.environment_reward_weight = environment_reward_weight
        self.environment_constraint_weight = environment_constraint_weight
        self.reference_dataset = np.load(reference_dataset_path)
        self.reference_length = len(self.reference_dataset["observations"])
        self.discriminator = Discriminator(
            self.reference_dataset,
            [512, 256],
            reward_shaping=reward_shaping,
            constraint_shaping=constraint_shaping,
            exponential_mean_discounting=discriminator_mean_discounting,
            imitation_reward_weight=imitation_reward_weight,
            imitation_constraint_weight=imitation_constraint_weight,
            imitation_constraint_slack=imitation_constraint_slack,
            optimizer=discriminator_optimizer, # type: ignore
            optimizer_kwargs=optimizer_kwargs,
            batch_size=discriminator_batch_size,
            weight_imitation=discriminator_loss_imiation_weight,
            weight_gradient_penalty=discriminator_loss_gradience_penalty_weight,
            gradient_steps=discriminator_training_steps,
            device=discriminator_device, # type: ignore
        )
        
    def _finish_env_step(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray, 
        actions: np.ndarray,
        info: Dict
    ):
        const_fn_eval = self.constraint_function(observations, muscle_states)
        info["const_fn_eval"] = const_fn_eval

        pred = self.discriminator.predict(
            self.agent.last_observations, # type: ignore
            observations,
        )
        discriminator_reward = self.discriminator.reward(
            pred,
        )
        discriminator_constraint = self.discriminator.constraint(
            pred,
        )
        self.discriminator.update_mean_and_var(pred)

        info["rewards"] *= self.environment_reward_weight
        info["rewards"] += discriminator_reward
        if self.environment_constraint_weight > 0:
            info["const_fn_eval"] *= self.environment_constraint_weight
        if not self.discriminator.imitation_constraint_weight <= 0:
            info["const_fn_eval"] = np.maximum(
                info["const_fn_eval"],
                discriminator_constraint
            )
        self._update_budget(info, info["const_fn_eval"])
        info["budgets"] = self._budgets.copy()

    def _discriminator_update_condition(self) -> bool:
        """
        Returns True, either if the has as many samples as the reference dataset
        or if the replay buffer is full
        """
        if self.agent.replay.index == 0: # type: ignore
            logger.store("imitation/discriminator_training/triggered_trough_replay_buffer", 1)
            logger.store("imitation/discriminator_training/triggered_trough_reference_dataset", 0)
            return True
        if self.agent.replay.index * self._num_workers % self.reference_length == 0: # type: ignore
            logger.store("imitation/discriminator_training/triggered_trough_replay_buffer", 0)
            logger.store("imitation/discriminator_training/triggered_trough_reference_dataset", 1)
            return True
        return False

    def _finish_update(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        super()._finish_update(observations, muscle_states, actions, info)
        if self._discriminator_update_condition():

            learn_dict = self.agent.replay.get_keys( # type: ignore
                "observations",
                "next_observations",
                "budgets",
            )

            budgets = learn_dict.pop("budgets")
            positive_budgets = budgets > 0

            learn_dict["observations"] = learn_dict["observations"][positive_budgets]
            learn_dict["next_observations"] = learn_dict["next_observations"][positive_budgets]

            self.discriminator.update(learn_dict)

    def _test(
        self,
    ):
        if not hasattr(self, "test_fn"):
            return
        self.test_fn(
            self.test_environment,
            self.agent,
            self.discriminator,
            self._steps,
            self.constraint_function,
            test_episodes = self.test_episodes,
            data_path = self.data_path,
        )
