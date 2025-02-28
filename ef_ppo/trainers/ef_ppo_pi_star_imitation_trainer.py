from typing import *
import numpy as np
from .ef_ppo_imitation_trainer import EFPPOImitationTrainer
from ef_ppo import logger
from time import time

class EFPPOPiStarImitationTrainer(EFPPOImitationTrainer):
    """
    EFPPOTrainer that supports imitation objectives
    """
    def __init__(
        self,
        pi_star_rollout_length: int = 32768,
        reset_discriminator_every: int = int(1e99),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pi_star_rollout_length = pi_star_rollout_length
        self.n_discriminator_updates = 0
        self.reset_discriminator_every = reset_discriminator_every
        
    def _discriminator_update_condition(self) -> bool:
        """
        Returns True, either if the has as many samples as the reference dataset
        or if the replay buffer is full
        """
        if self.agent.replay.index == 0: # type: ignore
            self.n_discriminator_updates += 1
            return True
        return False

    def _collect_pi_star_trajectories(
        self,
        num_transitions: int,
        observations: np.ndarray,
        muscle_states: np.ndarray,
    ) -> Dict[Literal["observations", "next_observations"], np.ndarray]:
        """
        Collects trajectories using the current optimal policy
        """
        time_start = time()
         
        observation_buffer = []
        next_observation_buffer = []
        for _ in range(num_transitions // self._num_workers + 1):
            actions = self.agent.test_step(observations, self._steps, muscle_states) # type: ignore
            observations, _, _ = self.environment.step(actions) # type: ignore
            observation_buffer.append(self.agent.last_observations.copy()) # type: ignore
            next_observation_buffer.append(observations.copy()) # type: ignore

        logger.store("imitation/discriminator_training/"
                     "collect_pi_star_trajectories_time",
                     time() - time_start)
        return {
            "observations": np.concatenate(observation_buffer),
            "next_observations": np.concatenate(next_observation_buffer),
        }

    def _finish_update(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        super(EFPPOImitationTrainer, self)._finish_update(
            observations,
            muscle_states,
            actions,
            info
        )
        if self.n_discriminator_updates % self.reset_discriminator_every == 0:
            self.discriminator.reset_regressor()
        if self._discriminator_update_condition():
            self.discriminator.update(
                self._collect_pi_star_trajectories(
                    self.pi_star_rollout_length,
                    observations,
                    muscle_states,
                )
            )
