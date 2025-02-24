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
        pi_star_rollout_length: int = 10000,
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
        num_transitions: int
    ) -> Dict[Literal["observations", "next_observations"], np.ndarray]:
        """
        Collects trajectories using the current optimal policy
        """
        # Start the environment if not already started
        time_start = time()
        if not hasattr(self.test_environment, "test_observations"):
            self.test_environment.test_observations, _ = self.test_environment.start() # type: ignore
            assert len(self.test_environment.test_observations) == 1 # type: ignore
         
        obs = self.test_environment.test_observations.copy() # type: ignore
        observations = []
        next_observations = []
        for _ in range(num_transitions):
            actions = self.agent.test_step(obs, self._steps) # type: ignore
            budget_star = self.agent.budget_star # type: ignore
            obs, _, _ = self.test_environment.step(actions) # type: ignore
            observations.append(self.agent.last_observations.flatten().copy()) # type: ignore
            next_observations.append(obs.flatten().copy()) # type: ignore

            # Log the budget_star
            logger.store("imitation/discriminator_training/budget_star", budget_star, stat_level="msM")

        logger.store("imitation/discriminator_training/"
                     "collect_pi_star_trajectories_time",
                     time() - time_start)
        return {
            "observations": np.array(observations),
            "next_observations": np.array(next_observations)
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
                    self.pi_star_rollout_length
                )
            )
