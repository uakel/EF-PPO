"""
Base Trainer Class
"""
# Typing
from typing import *
from deprl.vendor.tonic.agents import Agent
from deprl.custom_distributed import Parallel, Sequential
from ef_ppo.ef_ppo import EF_PPO

# Logging
from ef_ppo import logger
from .base_trainer import BaseTrainer

# Number Crunching
import numpy as np

class EFPPOTrainer(BaseTrainer):
    def __init__(
        self,
        constraint_function: str = "lambda obs, ms: -np.ones(obs.shape[0])",
        max_budget: float = 0,
        budget_update: Literal["paper", "modified", "sum", "none"] = "paper",
        max_violation: float = 0,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.constraint_function : Callable = eval(constraint_function)
        self.max_budget : float = max_budget
        self.budget_update : Literal["paper", "modified", "none", "sum"] = budget_update
        self.max_violation : float = max_violation

    def initialize(
        self, 
        agent: EF_PPO, 
        environment: Parallel | Sequential, 
        test_environment: Parallel | Sequential | None = None,
        full_save: bool = False
    ):
        super().initialize(agent, environment, test_environment, full_save)
        self.agent: EF_PPO = agent
        self.agent.max_budget = self.max_budget

    def _prepare_run(
            self,
            observations: np.ndarray,
            muscle_states: np.ndarray,
            num_workers: int
    ):
        self._budgets = np.random.uniform(low=0, high=self.max_budget, size=num_workers)
        self._constraint_returns = np.ones(num_workers, float) * -np.inf
        self._aleph = np.zeros(num_workers, float)
        super()._prepare_run(observations, muscle_states, num_workers)

    def _agent_step_args(
        self, 
        observations: np.ndarray,
        muscle_states: np.ndarray
    ) -> Tuple:
        return observations, self._steps, self._budgets, muscle_states

    def _paper_budget_update(
        self,
        info: Dict,
        constraint: np.ndarray
    ):
        rewards = info["rewards"]
        self._budgets = np.clip(
            (self._budgets + rewards) / self.discount,
            -self.max_budget,
             self.max_budget
        )
        
    def _modified_budget_update(
        self,
        info: Dict,
        constraint: np.ndarray
    ):
        self._budgets = np.clip(
            (
                self._budgets + info["rewards"] 
                + (1 - self.discount) * info["const_fn_eval"]
            ) / self.discount, 
            -self.max_budget, 
            self.max_budget
        ) 

    def _constant_budget_update(
        self,
        info: Dict,
        constraint: np.ndarray
    ):
        pass

    def _sum_budget_update(
        self,
        info: Dict,
        constraint: np.ndarray
    ):
        self._budgets = np.clip(
            (
                self._budgets + info["rewards"] 
                + info["const_fn_eval"]
                - (1 - self.discount) * self.max_violation
            ) / self.discount, 
            -self.max_budget, 
            self.max_budget
        ) 

    def _update_budget(
        self,
        info: Dict,
        constraint: np.ndarray
    ):
        if self.budget_update == "paper":
            self._paper_budget_update(info, constraint)
        elif self.budget_update == "modified":
            self._modified_budget_update(info, constraint)
        elif self.budget_update == "sum":
            self._sum_budget_update(info, constraint)
        else:
            self._constant_budget_update(info, constraint)

    def _finish_env_step(
        self, 
        observations: np.ndarray, 
        muscle_states: np.ndarray, 
        actions: np.ndarray, 
        info: Dict
    ):
        super()._finish_env_step(observations, muscle_states, actions, info)
        const_fn_eval = self.constraint_function(observations, muscle_states)
        info["const_fn_eval"] = const_fn_eval
        self._update_budget(info, const_fn_eval)
        info["budgets"] = self._budgets.copy()

    def _finish_update(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        super()._finish_update(observations, muscle_states, actions, info)
        ends = info["terminations"] | info["resets"]
        self._budgets[ends] = np.random.uniform(
            low=-self.max_budget,
            high=self.max_budget, 
            size=ends.sum()
        )
        self._constraint_returns = np.maximum(
            self._constraint_returns, 
            (1 - self.discount) * self._aleph 
            + self.discount ** self._lengths * info["const_fn_eval"]
        )
        self._aleph = self._aleph + self.discount ** self._lengths\
            * info["const_fn_eval"]

    def _test(
        self,
    ):
        if not hasattr(self, "test_fn"):
            return
        self.test_fn(
            self.test_environment,
            self.agent,
            self._steps,
            self.constraint_function,
            test_episodes = self.test_episodes,
            data_path = self.data_path,
        )

    def _end_episode(
        self,
        worker: int
    ):
        super()._end_episode(worker)
        logger.store(
            "train/constraint_return", 
            self._constraint_returns[worker], 
            stat_level="msM"
        )
        self._constraint_returns[worker] = -np.inf
        self._aleph[worker] = 0

    def _log_training_locals(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        super()._log_training_locals(observations, muscle_states, actions, info)
        logger.store("train/budgets", info["budgets"], stat_level="msM")
        logger.store("train/const_fn_eval", info["const_fn_eval"], stat_level="msM")



