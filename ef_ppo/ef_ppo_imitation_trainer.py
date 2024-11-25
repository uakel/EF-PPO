import os
import time

import numpy as np
import torch
from ef_ppo.discriminator import Discriminator
from ef_ppo import logger
from ef_ppo.ef_ppo import EF_PPO
from ef_ppo.utils import discounted_cost_score, discounted_constraint_score

if "ROBOHIVE_VERBOSITY" not in os.environ:
    os.environ["ROBOHIVE_VERBOSITY"] = "ALWAYS"

class ImitationTrainer:
    """
    EF-PPO Trainer class that utilizes imitation learning
    """
    def __init__(
        self,
        # Training routine parameters
        device="cuda",
        max_steps=1e7,
        epoch_steps=2e4,
        save_steps=2e4 * 5,
        test_episodes=20,

        # Logging and saving parameters
        show_progress=True,
        replace_checkpoint=False,
        data_path=lambda env: env.environments[0].sim.model,

        # Testing parameters
        test_hook = "ef_ppo.test_mujoco_with_muscles:test",

        # Constraint learning parameters
        max_budget=1,
        adaptive_budget=True,
        constraint_function="""
        lambda observations, muscle_states: -np.ones(
            len(observations)
        ).astype(np.float32)
        """, # Dummy constraint function 

        # MDP parameters
        discount=0.99,

        # Imitation learning parameters
        ## Imitation cost parameters
        standarize_discriminator_output=False,
        exponential_mean_discounting=0.9999,
        imitation_cost_multiplier=0.1,

        ## Reference dataset
        reference_dataset_path="dataset.npz",

        ## Discriminator Training
        discriminator_optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr" : 1e-4},
        weight_imitation=0.5,
        weight_gradient_penalty=0,
        discriminator_steps=8,
        update_frozen_discriminator_every=1,
        use_all_trajectories=False,
    ):
        # Saving the parameters
        # Training routine parameters
        self.device = device
        self.max_steps = int(max_steps)
        self.epoch_steps = int(epoch_steps)
        self.save_steps = int(save_steps)
        self.test_episodes = test_episodes

        # Logging and saving parameters
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint
        self.data_path = data_path
        
        # Testing parameters
        module_name, test_fn = test_hook.split(":")
        namespace = {}
        exec(f"from {module_name} import {test_fn} as evaluated_test_fn", namespace)
        self.test_fn = namespace["evaluated_test_fn"]

        # Constraint learning parameters
        self.max_budget = max_budget
        self.adaptive_budget = adaptive_budget
        self.constraint_function = eval(constraint_function)

        # MDP parameters
        self.discount = discount 

        # Imitation learning parameters and variables
        ## Reference dataset
        self.reference_dataset = np.load(reference_dataset_path)
        self.reference_length = len(self.reference_dataset["observations"])

        ## Discriminator
        self.discriminator = Discriminator(
            self.reference_dataset,
            [512, 256],
            standarize_output=standarize_discriminator_output,
            exponential_mean_discounting=exponential_mean_discounting,
            imitation_cost_multiplier=imitation_cost_multiplier,
            optimizer=discriminator_optimizer,
            optimizer_kwargs=optimizer_kwargs,
            weight_imitation=weight_imitation,
            weight_gradient_penalty=weight_gradient_penalty,
            gradient_steps=discriminator_steps,
            update_frozen_every=update_frozen_discriminator_every,
            device=device,
        )

        ## Discriminator Training
        self.use_all_trajectories = use_all_trajectories

    def initialize(
        self, agent, environment, 
        test_environment=None, 
        full_save=False
    ):
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def update_budget(self, costs, const_fn_evals):
        if self.adaptive_budget == "modified":
            self.budgets = np.clip(
                (self.budgets - costs) / self.discount,
                -self.max_budget,
                self.max_budget
            )
        elif type(self.adaptive_budget) == bool and self.adaptive_budget:
            self.budgets = np.clip(
                (self.budgets - costs + (1 - self.discount) * const_fn_evals)
                / self.discount, 
                -self.max_budget, 
                self.max_budget
            )
        logger.store("train/budgets", 
                     self.budgets, 
                     stats=True)

    def draw_new_budgets(self, episode_ends):
        self.budgets[episode_ends] = np.random.uniform(
            low=0 if self.adaptive_budget else -self.max_budget,
            high=self.max_budget,
            size=self.budgets[episode_ends].shape
        ).astype(np.float32)

    def create_logging_vars(self, observations, episodes, steps, epochs):
        self._num_workers = len(observations)
        self._cost_scores = np.zeros(self._num_workers)
        self._constraint_scores = np.zeros(self._num_workers)
        self._costs_since_reset = [[] for _ in range(self._num_workers)]
        self._constraint_fn_evals_since_reset = [[] for _ in range(self._num_workers)]
        self._lengths = np.zeros(self._num_workers, int)
        self._steps, self._epoch_steps = steps, 0
        self._steps_since_save = 0
        self._episodes = episodes
        self._epochs = epochs
        self._start_time = self._last_epoch_time = time.time()

    def update_logging_vars(self, info, const_fn_eval):
        self._cost_scores += info["costs"]
        self._constraint_scores = np.maximum(const_fn_eval, self._constraint_scores)
        for i in range(self._num_workers):
            self._costs_since_reset[i].append(info["costs"][i])
            self._constraint_fn_evals_since_reset[i].append(const_fn_eval[i])
        self._lengths += 1
        self._steps += self._num_workers
        self._epoch_steps += self._num_workers
        self._steps_since_save += self._num_workers

    def discriminator_update_condition(self):
        # Update the discriminator if the replay buffer is full or the reference length is reached.
        if self.agent.replay.long_term_buffer_index == 0:
            return False
        if self.agent.replay.index == 0:
            logger.store("imitation/discriminator_training/p_update_trigger_is_replay_full",
                         1)
            return True
        elif self.agent.replay.index * self._num_workers % self.reference_length == 0:
            logger.store("imitation/discriminator_training/p_update_trigger_is_replay_full",
                         0)
            return True
    
    def after_finished_episode(self, index):
        # Calculate discounted scores
        discounted_cost_scores = discounted_cost_score(
            self._costs_since_reset[index],
            self.discount
        )
        discounted_constraint_scores = discounted_constraint_score(
            self._constraint_fn_evals_since_reset[index], self.discount
        )
        
        # Store in Logger
        logger.store("train/cost/undiscounted_cost_score", self._cost_scores[index], stats=True)
        logger.store("train/constraint/undiscounted_constraint_score", self._constraint_scores[index], stats=True)
        for score in discounted_cost_scores:
            logger.store("train/cost/discounted_cost_score", score, stats=True)
        for score in discounted_constraint_scores:
            logger.store("train/constraint/discounted_constraint_score", score, stats=True)
        logger.store(
            "train/episode_length", self._lengths[index], stats=True
        )

        # Reset the variables.
        self._cost_scores[index] = 0
        self._constraint_scores[index] = 0
        self._costs_since_reset[index] = []
        self._constraint_fn_evals_since_reset[index] = []
        self._lengths[index] = 0
        self._episodes += 1

    def after_finished_epoch(self):
        # Update some quantities
        self._epochs += 1
        current_time = time.time()
        epoch_time = current_time - self._last_epoch_time
        sps = self._epoch_steps / epoch_time

        # Store in logger
        logger.store("train/episodes", self._episodes)
        logger.store("train/epochs", self._epochs)
        logger.store("train/seconds", current_time - self._start_time)
        logger.store("train/epoch_seconds", epoch_time)
        logger.store("train/epoch_steps", self._epoch_steps)
        logger.store("train/steps", self._steps)
        logger.store("train/worker_steps", self._steps // self._num_workers)
        logger.store("train/steps_per_second", sps)

        # Reset some variables
        self._last_epoch_time = time.time()
        self._epoch_steps = 0

        # Complete log
        logger.dump()

    def after_step(self, info, discriminator_cost, const_fn_eval, actions):
        # Update logging variables
        self.update_logging_vars(info, const_fn_eval)

        # Draw new budgets if termination or a reset has happened
        self.draw_new_budgets(info["terminations"] | info["resets"])

        # Store in Logger
        logger.store("train/cost/total_cost",
                     info["costs"], stats=True)
        logger.store("train/cost/discriminator_cost_fraction",
                     discriminator_cost / info["costs"], stats=True)
        logger.store("train/cost/environment_costs",
                     info["costs"] - discriminator_cost, stats=True)
        logger.store("train/constraint/constraint_function_evaluations", 
                     const_fn_eval, stats=True)
        logger.store("train/action",
                     actions, stats=True)

        # Show the progress bar.
        if self.show_progress:
            logger.show_progress(
                self._steps, self.epoch_steps, self.max_steps
            )

    def save_checkpoint(self):
        path = os.path.join(logger.get_path(), "checkpoints")
        if os.path.isdir(path) and self.replace_checkpoint:
            for file in os.listdir(path):
                if file.startswith("step_"):
                    os.remove(os.path.join(path, file))
        checkpoint_name = f"step_{self._steps}"
        save_path = os.path.join(path, checkpoint_name)
        # save agent checkpoint
        self.agent.save(save_path, full_save=False)
        # save logger checkpoint
        logger.save(save_path)
        # save time iteration dict
        self.save_time(save_path, self._epochs, self._episodes)
        self._steps_since_save = self._steps % self.save_steps

    def test(self, env, agent, steps, const_fn, test_params, data_path):
        self.test_fn(env,
                     agent, 
                     steps, 
                     const_fn, 
                     test_params, 
                     data_path=data_path, 
                     test_episodes=self.test_episodes)

    def run(self, test_params, steps=0, epochs=0, episodes=0, save=True):
        """
        Runs the main training loop.
        """
        # Start the environments.
        observations, muscle_states = self.environment.start()
        self.create_logging_vars(observations, episodes, steps, epochs)

        # Set max_budget
        self.agent.max_budget = self.max_budget

        # initialize the budgets
        self.budgets = np.zeros(self._num_workers, np.float32)
        self.draw_new_budgets(np.ones(self._num_workers, bool))

        # Start training loop
        while True:
            assert not np.isnan(observations.sum())
            
            # Get actions
            actions = self.agent.step(
                observations,
                self._steps,
                self.budgets,
                muscle_states,
            )
            assert not np.isnan(actions.sum())

            # Take a step in the environments.
            observations, muscle_states, info \
                = self.environment.step(actions)

            # Make rewards to costs
            info["costs"] = info.pop("rewards")

            # Remove env info 
            if "env_infos" in info:
                info.pop("env_infos")


            # Calculate and integrate discriminator cost
            discriminator_cost = self.discriminator.cost(
                self.agent.last_observations,
                observations
            )
            info["costs"] += discriminator_cost

            # Evaluate constraint function and get environment costs
            const_fn_eval = self.constraint_function(
                observations, 
                muscle_states
            )
            self.update_budget(info["costs"], const_fn_eval)

            # Update agent
            self.agent.update(
                **info, 
                const_fn_eval=const_fn_eval,
                budgets=self.budgets,
                steps=self._steps
            )
            
            # Update discriminator
            if self.discriminator_update_condition():
                if self.use_all_trajectories:
                    self.discriminator.update(
                        self.agent.replay.get_keys_from_entire_histroy("observations", "next_observations")
                    )
                else:
                    self.discriminator.update(
                        self.agent.replay.get_keys_from_current_segment("observations", "next_observations")
                    )

            # Finish the step
            self.after_step(info, discriminator_cost, const_fn_eval, actions)

            # Check the finished episodes.
            for index in range(self._num_workers):
                if info["resets"][index]:
                    self.after_finished_episode(index)
                    # Increase the number of episodes.
            
            # End of the epoch.
            if self._epoch_steps >= self.epoch_steps:
                # Evaluate the agent on the test environment.
                if self.test_environment:
                    self.test(self.test_environment,
                              self.agent,
                              self._steps,
                              self.constraint_function,
                              test_params,
                              data_path=self.data_path)
                self.after_finished_epoch()

            # End of training.
            stop_training = self._steps >= self.max_steps
            if stop_training or self._steps_since_save >= self.save_steps and save:
                # Save a checkpoint
                self.save_checkpoint()

            if stop_training:
                self.close_mp_envs()
                return self._cost_scores, self._constraint_scores, self._lengths, self._epochs, self._episodes

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
            "steps": self._steps,
        }
        torch.save(time_dict, time_path)

    def get_path(self, path, post_fix):
        return path.split("step")[0] + post_fix + ".pt"
