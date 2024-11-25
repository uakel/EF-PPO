import os
import time

import numpy as np
import torch
from ef_ppo.test_mujoco_with_muscles import test as test_mujoco

from ef_ppo import logger
from ef_ppo.ef_ppo import EF_PPO
from ef_ppo.utils import discounted_cost_score, discounted_constraint_score

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
        data_path=lambda env: env.environments[0].sim.model,
        adaptive_budget=True,
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
        self.data_path = data_path
        self.adaptive_budget = adaptive_budget

    def initialize(
        self, agent, environment, 
        test_environment=None, 
        full_save=False
    ):
        if not issubclass(type(agent), EF_PPO):
           logger.log("WARNING: You are using a non EF_PPO derived agent "
                       "together with the EF-PPO trainer.")

        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def determine_max_budget(self, init_observations, 
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
            costs = info["rewards"]
            episode_ends = info["terminations"] | info["resets"]

            # Return calculation
            returns += self.discount**steps_since_last_episode_end * costs

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
        
        # Create logging vars
        cost_scores = np.zeros(num_workers)
        constraint_scores = np.zeros(num_workers)
        costs_since_reset = [[] for _ in range(num_workers)]
        constraint_fn_evals_since_reset = [[] for _ in range(num_workers)]
        lengths = np.zeros(num_workers, int)
        self.steps, epoch_steps = steps, 0
        steps_since_save = 0

        # Start training loop
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
                muscle_states,
                greedy_episode=greedy_episode
            )

            assert not np.isnan(actions.sum())
            logger.store("train/action", actions, stats=True)

            # Take a step in the environments.
            observations, muscle_states, info \
                = self.environment.step(actions)

            # Make rewards to costs
            info["costs"] = info.pop("rewards")

            # Remove env info 
            if "env_infos" in info:
                info.pop("env_infos")

            # Get and log cost
            costs = info["costs"]
            logger.store("train/costs", costs, stats=True)

            # Evaluate constraint function and get environment costs
            const_fn_eval = self.constraint_function(
                observations, 
                muscle_states
            )
            if self.adaptive_budget == "modified":
                budgets = np.clip((budgets - costs) / self.discount,
                                  -self.max_budget,
                                  self.max_budget)
            elif type(self.adaptive_budget) == bool and self.adaptive_budget:
                budgets = np.clip((budgets - costs 
                                   + (1 - self.discount) * const_fn_eval)
                                  / self.discount, -self.max_budget, self.max_budget)

            # Store in logger
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
            budgets[episode_ends] = np.random.uniform(
                low= 0 if self.adaptive_budget else -self.max_budget,
                high=self.max_budget, 
                size=budgets[episode_ends].shape
            ).astype(np.float32)

            # Update logging variables
            cost_scores += info["costs"]
            constraint_scores = np.maximum(const_fn_eval, constraint_scores)
            for i in range(num_workers):
                costs_since_reset[i].append(info["costs"][i])
                constraint_fn_evals_since_reset[i].append(const_fn_eval[i])

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
                    # Store undiscounted scores.
                    logger.store("train/cost_score", cost_scores[i], stats=True)
                    logger.store("train/constraint_score", constraint_scores[i],
                                 stats=True)
                    # Store the discounted scores.
                    discounted_cost_scores = discounted_cost_score(
                        costs_since_reset[i],
                        self.discount
                    )
                    for score in discounted_cost_scores:
                        logger.store("train/discounted_cost_score", score, stats=True)

                    # Store the discounted constraint scores.
                    discounted_constraint_scores = discounted_constraint_score(
                        constraint_fn_evals_since_reset[i], self.discount
                    )
                    for score in discounted_constraint_scores:
                        logger.store("train/discounted_constraint_score", score, stats=True)
                    
                    # Store the episode length.
                    logger.store(
                        "train/episode_length", lengths[i], stats=True
                    )
                    # Reset the variables.
                    cost_scores[i] = 0
                    constraint_scores[i] = 0
                    costs_since_reset[i] = []
                    constraint_fn_evals_since_reset[i] = []
                    lengths[i] = 0
                    # Increase the number of episodes.
                    episodes += 1
            
            # End of the epoch.
            if epoch_steps >= self.epoch_steps:
                # Evaluate the agent on the test environment.
                if self.test_environment:
                    _ = test_mujoco(self.test_environment,
                                    self.agent,
                                    steps,
                                    self.constraint_function,
                                    params,
                                    data_path=self.data_path)

                # Update some quantities
                epochs += 1
                current_time = time.time()
                epoch_time = current_time - last_epoch_time
                sps = epoch_steps / epoch_time

                # Store in logger
                logger.store("train/episodes", episodes)
                logger.store("train/epochs", epochs)
                logger.store("train/seconds", current_time - start_time)
                logger.store("train/epoch_seconds", epoch_time)
                logger.store("train/epoch_steps", epoch_steps)
                logger.store("train/steps", self.steps)
                logger.store("train/worker_steps", self.steps // num_workers)
                logger.store("train/steps_per_second", sps)

                # Reset some variables
                last_epoch_time = time.time()
                epoch_steps = 0

                # Complete log
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
                return cost_scores, constraint_scores, lengths, epochs, episodes


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
