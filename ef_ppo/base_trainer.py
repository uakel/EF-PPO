"""
Base Trainer Class
"""
# Typing
from typing import *
from deprl.vendor.tonic.agents import Agent
from deprl.custom_distributed import Parallel, Sequential

# Logging
import os
import time
from ef_ppo import logger

# Number Crunching
import numpy as np
import torch

# Trainer Class
class BaseTrainer:
    """
    Base Trainer Class
    """
    def __init__(
        self,
        steps : int               = int(4096 * 16 * 4 * 10 * 256),
        epoch_steps : int         = int(4096 * 16 * 4 * 10),
        save_steps : int          = int(4096 * 16 * 4 * 10),
        test_episodes : int       = 20,
        test_fn : str | None      = None, # "module_name:fn_name"
        discount: float           = 0.99,
        show_progress : bool      = True,
        data_path : Callable      = lambda env: env.environments[0].unwrapped.sim.data,
    ): 
        """
        Note: test_fn has to have the following signature:
            test_environment: Parallel | Sequential, 
            agent: Agent, 
            steps: int, 
            data_path: Callable, 
            test_episodes: int
        """
        # Save the parameters
        self.max_steps : int = int(steps)
        self.epoch_steps : int = int(epoch_steps)
        self.save_steps : int = int(save_steps)
        self.test_episodes : int = test_episodes
        self.discount : float = discount 
        self.show_progress = show_progress
        self.data_path = data_path

        # Load the test function
        if test_fn is not None:
            module_name, test_fn = test_fn.split(":")
            namespace = {}
            exec(f"from {module_name} import {test_fn} as evaluated_test_fn", namespace)
            self.test_fn = namespace["evaluated_test_fn"]

    def initialize(
        self, 
        agent : Agent,
        environment : Parallel | Sequential,
        test_environment : Parallel | Sequential | None = None,
        full_save : bool                                = False,
    ):
        self.agent : Agent = agent
        self.environment : Parallel | Sequential = environment
        self.test_environment : Parallel | Sequential | None = test_environment
        self.full_save : bool = full_save

    def prepare_run(
            self,
            observations: np.ndarray, 
            muscle_states: np.ndarray,
            num_workers: int,
        ):
        pass

    def finish_agent_step(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        pass

    def agent_step_args(
            self, 
            observations: np.ndarray,
            muscle_states: np.ndarray
    ) -> Tuple:
        return observations, self._steps

    def agent_update_args(
            self,
            info: Dict,
            steps: int
    ) -> Dict:
        info["steps"] = steps
        return info

    def finish_env_step(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        pass

    def finish_update(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        # Update monitoring variables
        self._lengths += 1
        self._steps += self._num_workers
        self._steps_in_curr_epoch += self._num_workers
        self._steps_since_save += self._num_workers
        self._returns += self.discount**self._lengths * info["rewards"]

        # Log the local variables generated in the training loop
        self.log_training_locals(observations, muscle_states, actions, info)

        # Print the progress bar
        if self.show_progress:
            logger.show_progress(
                self._steps, self.epoch_steps, self.max_steps
            )

    def end_episode(
        self,
        worker: int,
    ):
        logger.store("train/episode_length", self._lengths[worker], stat_level="msM")
        logger.store("train/episode_return", self._returns[worker], stat_level="msM")

        self._lengths[worker] = 0
        self._returns[worker] = 0

        self._episodes += 1

    def test(
        self,
    ):
        if not hasattr(self, "test_fn"):
            return
        self.test_fn(
            self.test_environment,
            self.agent,
            self._steps,
            test_episodes = self.test_episodes,
            data_path = self.data_path,
        )

    def log_epoch_statistics(
        self,
    ):
        logger.store("train/episodes", self._episodes)
        logger.store("train/epochs", self._epochs)
        logger.store("train/seconds", time.time() - self._start_time)
        logger.store("train/epoch_seconds", time.time() - self._last_epoch_time)
        logger.store("train/epoch_steps", self._steps_in_curr_epoch)
        logger.store("train/steps", self._steps)
        logger.store("train/worker_steps", self._steps // self._num_workers)
        logger.store("train/steps_per_second", 
                     self._steps / (time.time() - self._last_epoch_time))

    def end_epoch(
        self,
    ):
        if self.test_environment is not None:
            self.test()

        self._epochs += 1
        self.log_epoch_statistics()
        logger.dump()
        self._last_epoch_time = time.time()
        self._steps_in_curr_epoch = 0

    def close_mp_envs(self):
        for index in range(len(self.environment.processes)): # type: ignore
            self.environment.processes[index].terminate()    # type: ignore
            self.environment.action_pipes[index].close()     # type: ignore
        self.environment.output_queue.close()                # type: ignore    

    def save_time(self):
        time_path = os.path.join(logger.get_path(), "checkpoints/time.pt")
        time_dict = {
            "epochs": self._epochs,
            "episodes": self._episodes,
            "steps": self._steps,
        }
        torch.save(time_dict, time_path)

    def end_training(
        self,
    ):
        self.save_checkpoint()
        self.close_mp_envs()
        self.save_time()

    def save_checkpoint(
        self,
    ):
        path = os.path.join(logger.get_path(), "checkpoints")
        checkpoint_name = f"step_{self._steps}"
        save_path = os.path.join(path, checkpoint_name)
        if self._save:
            # save agent checkpoint
            self.agent.save(save_path, full_save=self.full_save)
            # save logger checkpoint
            logger.save(save_path)
            # save time iteration dict
            self.save_time()
            self._steps_since_save = self._steps % self.save_steps

    def log_training_locals(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        logger.store("train/action", actions, stat_level="msM")
        logger.store("train/rewards", info["rewards"], stat_level="msM")

    def run(
            self, 
            params : Dict, 
            steps : int = 0,
            epochs : int = 0, 
            episodes : int = 0,
            save : bool =True
    ):
        # Save the parameters
        self._params = params
        self._steps = steps
        self._epochs = epochs
        self._episodes = episodes
        self._save = save

        # Start the environments.
        observations, muscle_states = self.environment.start()
        info = {}
        self._num_workers = len(observations)
        
        # Create Logging Variables
        self._start_time = self._last_epoch_time = time.time()
        self._lengths = np.zeros(self._num_workers, int)
        self._steps_in_curr_epoch = 0
        self._steps_since_save = 0
        self._returns = np.zeros(self._num_workers, float)

        # Call preparation hook
        self.prepare_run(observations, muscle_states, self._num_workers)

        # Start training loop
        while True:
            # Get actions
            actions = self.agent.step(*self.agent_step_args(observations, muscle_states))
            self.finish_agent_step(observations, muscle_states, actions, info) # type: ignore
            # Take a step in the environments.
            observations, muscle_states, info = self.environment.step(actions)
            self.finish_env_step(observations, muscle_states, actions, info) # type: ignore

            # Update the agent
            # Update agent
            self.agent.update(**self.agent_update_args(info, self._steps))
            self.finish_update(observations, muscle_states, actions, info) # type: ignore

            # Handle episode termination workloads
            for w in range(self._num_workers):
                if info["resets"][w]:
                    self.end_episode(w)

            # End of epoch
            if self._steps_in_curr_epoch >= self.epoch_steps:
                self.end_epoch()

            # Save a checkpoint
            if self._steps_since_save >= self.save_steps:
                self.save_checkpoint()
                self._steps_since_save = 0

            # End of training
            if self._steps >= self.max_steps:
                self.end_training()
                break


class EFPPOTrainer(BaseTrainer):
    def __init__(
        self,
        constraint_function: str = "lambda obs, ms: -np.ones(obs.shape[0])",
        max_budget: float = 0,
        budget_update: Literal["paper", "modified", "none"] = "paper",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.constraint_function : Callable = eval(constraint_function)
        self.max_budget : float = max_budget
        self.budget_update : Literal["paper", "modified", "none"] = budget_update

    def initialize(
        self, 
        agent: Agent, 
        environment: Parallel | Sequential, 
        test_environment: Parallel | Sequential | None = None,
        full_save: bool = False
    ):
        super().initialize(agent, environment, test_environment, full_save)
        self.agent.max_budget = self.max_budget

    def prepare_run(
            self,
            observations: np.ndarray,
            muscle_states: np.ndarray,
            num_workers: int
    ):
        self._budgets = np.random.uniform(low=0, high=self.max_budget, size=num_workers)
        self._constraint_returns = np.ones(num_workers, float) * -np.inf
        self._aleph = np.zeros(num_workers, float)
        super().prepare_run(observations, muscle_states, num_workers)

    def agent_step_args(
        self, 
        observations: np.ndarray,
        muscle_states: np.ndarray
    ) -> Tuple:
        return observations, self._steps, self._budgets, muscle_states

    def paper_budget_update(
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
        
    def modified_budget_update(
        self,
        info: Dict,
        constraint: np.ndarray
    ):
        rewards = info["rewards"]
        self._budgets = np.clip(
            (
                self._budgets + info["rewards"] 
                + (1 - self.discount) * info["const_fn_eval"]
            ) / self.discount, 
            -self.max_budget, 
            self.max_budget
        ) 

    def constant_budget_update(
        self,
        info: Dict,
        constraint: np.ndarray
    ):
        pass

    def update_budget(
        self,
        info: Dict,
        constraint: np.ndarray
    ):
        if self.budget_update == "paper":
            self.paper_budget_update(info, constraint)
        elif self.budget_update == "modified":
            self.modified_budget_update(info, constraint)
        else:
            self.constant_budget_update(info, constraint)

    def finish_env_step(
        self, 
        observations: np.ndarray, 
        muscle_states: np.ndarray, 
        actions: np.ndarray, 
        info: Dict
    ):
        super().finish_env_step(observations, muscle_states, actions, info)
        const_fn_eval = self.constraint_function(observations, muscle_states)
        info["const_fn_eval"] = const_fn_eval
        self.update_budget(info, const_fn_eval)
        info["budgets"] = self._budgets.copy()

    def finish_update(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        super().finish_update(observations, muscle_states, actions, info)
        ends = info["terminations"] | info["resets"]
        self._budgets[ends] = np.random.uniform(
            low=-self.max_budget if self.budget_update == "none" else 0,
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

    def test(
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

    def end_episode(
        self,
        worker: int
    ):
        super().end_episode(worker)
        logger.store(
            "train/constraint_return", 
            self._constraint_returns[worker], 
            stat_level="msM"
        )
        self._constraint_returns[worker] = -np.inf
        self._aleph[worker] = 0

    def log_training_locals(
        self,
        observations: np.ndarray,
        muscle_states: np.ndarray,
        actions: np.ndarray,
        info: Dict,
    ):
        super().log_training_locals(observations, muscle_states, actions, info)
        logger.store("train/budgets", info["budgets"], stat_level="msM")
        logger.store("train/const_fn_eval", info["const_fn_eval"], stat_level="msM")



