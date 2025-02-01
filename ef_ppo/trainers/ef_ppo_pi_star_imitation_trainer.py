from typing import *
import numpy as np
import torch
from .ef_ppo_imitation_trainer import EFPPOImitationTrainer
from ef_ppo.discriminator import Discriminator
from ef_ppo import logger

class EFPPOPiStarImitationTrainer(EFPPOImitationTrainer):
    """
    EFPPOTrainer that supports imitation objectives
    """
    def __init__(
        self,
        pi_star_rollout_length: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pi_star_rollout_length = pi_star_rollout_length
        
    def _discriminator_update_condition(self) -> bool:
        """
        Returns True, either if the has as many samples as the reference dataset
        or if the replay buffer is full
        """
        if self.agent.replay.index == 0: # type: ignore
            logger.store("imitation/discriminator_training/triggered_trough_replay_buffer", 1)
            logger.store("imitation/discriminator_training/triggered_trough_reference_dataset", 0)
            return True
        return False

    def _collect_pi_star_trajectories(
        self,
        num_transitions: int
    ) -> Dict[str, np.ndarray]:
        """
        Collects trajectories using the current optimal policy

        Args:
            num_transitions: number of transitions to collect

        Returns:
            A dictionary containing the following keys:
            - observations: np.ndarray
            - next_observations: np.ndarray
        """
        # Start the environment if not already started
        if not hasattr(self.test_environment, "test_observations"):
            self.test_environment.test_observations, _ = self.test_environment.start() # type: ignore
            assert len(self.test_environment.test_observations) == 1 # type: ignore
         
        obs = self.test_environment.test_observations.copy() # type: ignore
        observations = []
        next_observations = []
        for _ in range(num_transitions):
            actions, budget_star = self.agent.test_step(obs, self._steps) # type: ignore
            obs, _, _ = self.test_environment.step(actions) # type: ignore
            observations.append(agent.last_observations) # type: ignore
            next_observations.append(obs)

            # Log the budget_star
            logger.store("imitation/discriminator_training/budget_star", budget_star, stat_level="msM")

        return dict(
            observations=np.array(observations),
            next_observations=np.array(next_observations)
        )

    def _finish_update(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        super()._finish_update(observations, muscle_states, actions, info)
        if self._discriminator_update_condition():
            self.discriminator.update(
                self._collect_pi_star_trajectories(
                    self.pi_star_rollout_length
                )
            )
