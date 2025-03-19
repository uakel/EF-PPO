from typing import Dict
import numpy as np
from .ef_ppo_trainer import EFPPOTrainer

class ActivatingRewardEFPPOTrainer(EFPPOTrainer):
    def __init__(
        self,
        tau: float = 25e6,
        max_scale: float = 1 / 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tau = tau
        self.max_scale = max_scale

    def _finish_env_step(
        self, 
        observations: np.ndarray, 
        muscle_states: np.ndarray, 
        actions: np.ndarray, 
        info: Dict
    ):

        super()._finish_env_step(observations, muscle_states, actions, info)
        scale = 1 - np.exp(-self._steps / self.tau)
        penalty = info["rewards"].copy()
        info["rewards"] = (1 - scale * penalty) * self.max_scale

