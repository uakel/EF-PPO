import os
import time

import numpy as np
import torch

from deprl.custom_test_environment import (
    test_dm_control,
    test_scone,
)
from ef_ppo.test_environment import test_mujoco

from deprl.vendor.tonic import logger
from ef_ppo.ef_ppo import EF_PPO

if "ROBOHIVE_VERBOSITY" not in os.environ:
    os.environ["ROBOHIVE_VERBOSITY"] = "ALWAYS"

class Trainer:
    """
    EF-PPO Trainer class
    """
    def __init__(
        self,
        steps=1e7,
        epoch_steps=2e4,
        save_steps=5e5,
        test_episodes=20,
        discount=0.99,
        constraint_function="""
        lambda observations, muscle_states: -np.ones(
            len(observations)
        ).astype(np.float32)
        """, # Dummy constraint function 
        show_progress=True,
        replace_checkpoint=False,
        max_budget=None, 
    ):
        # Saving the parameters
        self.max_steps = int(steps)
        self.epoch_steps = int(epoch_steps)
        self.save_steps = int(save_steps)
        self.test_episodes = test_episodes
        self.discount = discount 
        self.constraint_function = eval(constraint_function)
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint
        self.max_budget = max_budget
    

    def determine_max_budget(self, agent, environment, init_observations, 
                             init_muscle_states, num_workers, n_episodes=100):
        # Log 
        logger.log("Max budget not given. "
                   "Determining max budget empirically..") 

        # Initialize some vars
        performed_episodes = 0
        returns = np.zeros(num_workers)
        observations = init_observations
        muscle_states = init_muscle_states
        steps = 0
        steps_since_last_episode_end = np.zeros(num_workers)
        max_return = 0

        # Calculate empirical returns
        while performed_episodes <= n_episodes:
            # Get actions
            actions = self.agent.step(
                observations,
                steps,
                np.zeros(num_workers).astype(np.float32),
                muscle_states
            )

            # Take environment step 
            observations, muscle_states, info \
                = self.environment.step(actions)
            losses = info["rewards"]
            episode_ends = info["terminations"] | info["resets"]

            # Return calculation
            returns += self.discount**steps_since_last_episode_end * losses

            # Increase index variables
            steps += 1
            steps_since_last_episode_end += 1

            # Handle ending episodes
            steps_since_last_episode_end[episode_ends] = 0
            performed_episodes += np.sum(episode_ends)
            if np.sum(episode_ends) > 0: 
                max_return = max(max_return, max(returns[episode_ends]))
            returns[episode_ends] = 0

        logger.log(f"Max budget determined as: {max_return}")
        return max_return


    def initialize(
        self, agent, environment, 
        test_environment=None, full_save=False
    ):
        if not issubclass(type(agent), EF_PPO):
           logger.log("WARNING: You are using a non EF_PPO derived agent "
                       "together with the EF-PPO trainer.")

        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment


    def run(self, params, steps=0, epochs=0, episodes=0, save=True):
        """
        Runs the main training loop.
        """

        # Start the environments.
        observations, muscle_states = self.environment.start()
        num_workers = len(observations)

        # Determine max budget empricially if not given
        if self.max_budget is None:
            self.max_budget = self.determine_max_budget(
                self.agent,
                self.environment,
                observations,
                muscle_states,
                num_workers
            )

        # Set max_budget
        self.agent.max_budget = self.max_budget

        # Start the actual training
        start_time = last_epoch_time = time.time()
        
        budgets = np.random.uniform(low=0, 
                                    high=self.max_budget,
                                    size=num_workers).astype(np.float32)

        scores = np.zeros(num_workers)
        constraint_score = np.zeros(num_workers)
        lengths = np.zeros(num_workers, int)
        self.steps, epoch_steps = steps, 0
        steps_since_save = 0

        while True:
            # Check greedy episode
            if hasattr(self.agent, "expl"):
                greedy_episode = (
                    not episodes % self.agent.expl.test_episode_every
                )
            else:
                greedy_episode = None

            # Check nan observations
            assert not np.isnan(observations.sum())
            
            # Get actions
            actions = self.agent.step(
                observations,
                self.steps,
                budgets,
                muscle_states
            )

            assert not np.isnan(actions.sum())
            logger.store("train/action", actions, stats=True)

            # Take a step in the environments.
            observations, muscle_states, info \
                = self.environment.step(actions)
            info["losses"] = info.pop("rewards")
            if "env_infos" in info:
                info.pop("env_infos")

            # Evaluate constraint function and get environment loss
            losses = info["losses"]
            const_fn_eval = self.constraint_function(
                observations, 
                muscle_states
            )
            budgets = np.clip((budgets - losses 
                               + (1 - self.discount) * const_fn_eval)
                              / self.discount, -self.max_budget, self.max_budget)
            logger.store("train/constraint_function_evaluations", 
                         const_fn_eval, stats=True)
            logger.store("train/budgets", 
                         budgets, stats=True)

            # Update agent
            self.agent.update(
                **info, 
                const_fn_eval=const_fn_eval,
                budgets=budgets,
                steps=self.steps
            )
            
            # Draw new budgets if termination or a reset has happened
            episode_ends = info["terminations"] | info["resets"]
            n_ends = np.sum(episode_ends)
            budgets[episode_ends] = np.random.uniform(
                low=0,
                high=self.max_budget, 
                size=budgets[episode_ends].shape
            ).astype(np.float32)

            # Log
            scores += info["losses"]
            constraint_score += const_fn_eval
            lengths += 1
            self.steps += num_workers
            epoch_steps += num_workers
            steps_since_save += num_workers

            # Show the progress bar.
            if self.show_progress:
                logger.show_progress(
                    self.steps, self.epoch_steps, self.max_steps
                )

            # Check the finished episodes.
            for i in range(num_workers):
                if info["resets"][i]:
                    logger.store("train/episode_score", scores[i], stats=True)
                    logger.store("train/constraint_score", constraint_score[i],
                                 stats=True)
                    logger.store(
                        "train/episode_length", lengths[i], stats=True
                    )
                    if i == 0:
                        # adaptive energy cost
                        if hasattr(self.agent.replay, "action_cost"):
                            logger.store(
                                "train/action_cost_coeff",
                                self.agent.replay.action_cost,
                            )
                            self.agent.replay.adjust(scores[i])
                    scores[i] = 0
                    constraint_score[i] = 0
                    lengths[i] = 0
                    episodes += 1

            # End of the epoch.
            if epoch_steps >= self.epoch_steps:
                # Evaluate the agent on the test environment.
                if self.test_environment:
                    _ = test_mujoco(self.test_environment,
                                    self.agent,
                                    steps,
                                    self.constraint_function,
                                    params)

                # Log the data.
                epochs += 1
                current_time = time.time()
                epoch_time = current_time - last_epoch_time
                sps = epoch_steps / epoch_time
                logger.store("train/episodes", episodes)
                logger.store("train/epochs", epochs)
                logger.store("train/seconds", current_time - start_time)
                logger.store("train/epoch_seconds", epoch_time)
                logger.store("train/epoch_steps", epoch_steps)
                logger.store("train/steps", self.steps)
                logger.store("train/worker_steps", self.steps // num_workers)
                logger.store("train/steps_per_second", sps)
                last_epoch_time = time.time()
                epoch_steps = 0

                logger.dump()

            # End of training.
            stop_training = self.steps >= self.max_steps

            # Save a checkpoint.
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(logger.get_path(), "checkpoints")
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith("step_"):
                            os.remove(os.path.join(path, file))
                checkpoint_name = f"step_{self.steps}"
                save_path = os.path.join(path, checkpoint_name)
                if save:
                    # save agent checkpoint
                    self.agent.save(save_path, full_save=False)
                    # save logger checkpoint
                    logger.save(save_path)
                    # save time iteration dict
                    self.save_time(save_path, epochs, episodes)
                    steps_since_save = self.steps % self.save_steps
                current_time = time.time()

            if stop_training:
                self.close_mp_envs()
                return scores


    def close_mp_envs(self):
        for index in range(len(self.environment.processes)):
            self.environment.processes[index].terminate()
            self.environment.action_pipes[index].close()
        self.environment.output_queue.close()


    def save_time(self, path, epochs, episodes):
        time_path = self.get_path(path, "time")
        time_dict = {
            "epochs": epochs,
            "episodes": episodes,
            "steps": self.steps,
        }
        torch.save(time_dict, time_path)


    def get_path(self, path, post_fix):
        return path.split("step")[0] + post_fix + ".pt"
