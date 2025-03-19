from typing import Dict
import numpy as np
from .ef_ppo_trainer import EFPPOTrainer

class DecayingConstraintEFPPOTrainer(EFPPOTrainer):
    def __init__(
        self,
        tau: float = 25e6,
        slack: float = 0,
        start: float = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tau = tau
        self.slack = slack
        self.start = start

    def _finish_env_step(
        self, 
        observations: np.ndarray, 
        muscle_states: np.ndarray, 
        actions: np.ndarray, 
        info: Dict
    ):
        super(EFPPOTrainer, self)._finish_env_step(observations, muscle_states, actions, info)
        if self.use_env_constraint:
            const_fn_eval = info["constraint"]
        else:
            const_fn_eval = self.constraint_function(observations, muscle_states) # type: ignore
        info.pop("constraint")
        treshold = (self.start - self.slack) * np.exp(-self._steps / self.tau) + self.slack
        const_fn_eval -= treshold
        info["const_fn_eval"] = const_fn_eval.copy()
        self._update_budget(info, const_fn_eval)
        info["budgets"] = self._budgets.copy()
