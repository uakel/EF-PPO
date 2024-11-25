import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from ef_ppo.test_environment import test_mujoco

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
        steps=1e7,
        epoch_steps=2e4,
        save_steps=5e5,
        test_episodes=20,
        discount=0.99,
        imitation_cost_alpha=0.9999,
        constraint_function="""
        lambda observations, muscle_states: -np.ones(
            len(observations)
        ).astype(np.float32)
        """, # Dummy constraint function 
        show_progress=True,
        test_mode="mujoco_w_muscles",
        replace_checkpoint=False,
        max_budget=1,
        data_path=lambda env: env.environments[0].sim.model,
        adaptive_budget=True,
        reference_dataset_path="dataset.npz",
        discriminator_optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr" : 1e-4},
        weight_imitation=0.5,
        weight_gradient_penalty=5,
        imitation_cost_multiplier=0.1,
        device="cuda",
        discriminator_steps=8,
        standarize_discriminator_output=False,
        update_discriminator_every=16,
        enable_gradient_penalty=False,
        all_trajectories=False,
    ):
        # Saving the parameters
        # Training routine parameters
        self.max_steps = int(steps)
        self.epoch_steps = int(epoch_steps)
        self.save_steps = int(save_steps)
        self.test_episodes = test_episodes
        self.max_budget = max_budget

        # Logging and saving parameters
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint
        self.data_path = data_path
        self.adaptive_budget = adaptive_budget
        self.test_mode = test_mode

        # MDP parameters
        self.discount = discount 
        self.constraint_function = eval(constraint_function)

        # Imitation learning parameters and variables
        self.imitation_cost_alpha = imitation_cost_alpha
        self.current_imitation_cost_mean_estimate = 0
        self.current_imitation_cost_var_estimate = 0
        self.reference_dataset = np.load(reference_dataset_path)
        self.reference_length = len(self.reference_dataset["observations"])
        self.device = device
        observation_dimension = self.reference_dataset["observations"].shape[1]
        self.discriminator = self.Discriminator(
            [observation_dimension * 2, 512, 256, 1]
        )
        self.discriminator_optimizer = discriminator_optimizer(
            self.discriminator.parameters(), **optimizer_kwargs
        )
        self.discriminator.to(self.device)
        self.frozen_discriminator = self.Discriminator(
            [observation_dimension * 2, 512, 256, 1]
        ).to(self.device)
        self.frozen_discriminator.load_state_dict(self.discriminator.state_dict())

        self.weight_imitation = weight_imitation
        self.weight_gradient_penalty = weight_gradient_penalty
        self.imitation_cost_multiplier = imitation_cost_multiplier
        self.discriminator_running_mean_and_var = np.zeros(2)
        def mean_and_var_update(x, y):
            add = np.array([y, (y - x[0]) ** 2])
            return self.imitation_cost_alpha * x + (1 - self.imitation_cost_alpha) * add

        self.mean_and_var_update = np.frompyfunc(mean_and_var_update, 2, 1)
        self.standarize_discriminator_output = standarize_discriminator_output
        self.discriminator_steps = discriminator_steps
        self.n_discriminator_updates = 0
        self.update_discriminator_every = update_discriminator_every
        self.enable_gradient_penalty = enable_gradient_penalty
        self.all_trajectories = all_trajectories

    def initialize(
        self, agent, environment, 
        test_environment=None, 
        full_save=False
    ):
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    class Discriminator(torch.nn.Module):
        def __init__(self, dims, activation=torch.nn.ReLU):
            super().__init__()
            layers = []
            for this_dim, next_dim in zip(dims[:-1], dims[1:]):
                layers.append(torch.nn.Linear(this_dim, next_dim))
                layers.append(activation())
            layers.pop()
            self.model = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    def data_iterator(self, learner_dataset, batch_size=256):
        shortest = min(self.reference_length, 
                       len(learner_dataset["observations"]))
        reference_indices = np.random.choice(
            len(self.reference_dataset["observations"]), shortest, replace=False
        )
        learner_indices = np.random.choice(
            len(learner_dataset["observations"]), shortest, replace=False
        )
        for i in range(0, shortest, batch_size):
            reference = np.concatenate(
                [
                    self.reference_dataset["observations"][reference_indices[i:i+batch_size]],
                    self.reference_dataset["next_observations"][reference_indices[i:i+batch_size]]
                ],
                axis=1
            )
            learner = np.concatenate(
                [
                    learner_dataset["observations"][learner_indices[i:i+batch_size]],
                    learner_dataset["next_observations"][learner_indices[i:i+batch_size]]
                ],
                axis=1
            )
            yield (torch.tensor(reference, dtype=torch.float32).to(self.device),
                   torch.tensor(learner, dtype=torch.float32).to(self.device))

    def train_discriminator(self, learner_dataset, epochs=1, batch_size=128):
        self.n_discriminator_updates += 1
        for _ in range(epochs):
            confusion_matrix = np.zeros((2, 2))
            it = 0
            for reference, learner in self.data_iterator(learner_dataset, batch_size):
                if it >= self.discriminator_steps:
                    break
                self.discriminator_optimizer.zero_grad()

                reference.requires_grad = True
                pred_reference = self.discriminator(reference)
                grad = torch.autograd.grad(
                    outputs=pred_reference,
                    inputs=reference,
                    grad_outputs=torch.ones_like(pred_reference),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

                if self.enable_gradient_penalty:
                    gradient_penalty = torch.mean(torch.norm(grad, dim=1) ** 2)

                pred_learner = self.discriminator(learner)

                loss = self.weight_imitation\
                     * F.binary_cross_entropy_with_logits(pred_learner, 
                                                          torch.zeros_like(pred_learner)) \
                     + self.weight_imitation\
                     * F.binary_cross_entropy_with_logits(pred_reference, 
                                                          torch.ones_like(pred_reference)) \
                     + (self.weight_gradient_penalty * gradient_penalty
                        if self.enable_gradient_penalty else 0)
                # -> minimize pred_learner, maximize pred_reference
                loss.backward()
                self.discriminator_optimizer.step()

                with torch.no_grad():
                    confusion_matrix[0, 0] += (pred_learner <= 0).sum().item()
                    confusion_matrix[0, 1] += (pred_learner >= 0).sum().item()
                    confusion_matrix[1, 0] += (pred_reference < 0).sum().item()
                    confusion_matrix[1, 1] += (pred_reference > 0).sum().item()

                    logger.store("imitation/discriminator_training/pred_learner/mean", 
                                 pred_learner.mean().item())
                    logger.store("imitation/discriminator_training/pred_learner/std", 
                                 pred_learner.std().item())
                    logger.store("imitation/discriminator_training/pred_reference/mean",
                                 pred_reference.mean().item())
                    logger.store("imitation/discriminator_training/pred_reference/std",
                                 pred_reference.std().item())
                    logger.store("imitation/discriminator_training/loss/total", 
                                 loss.item())
                    if self.enable_gradient_penalty:
                        logger.store("imitation/discriminator_training/loss/gradient_penalty",
                                     gradient_penalty.item() * self.weight_gradient_penalty)
                        logger.store("imitation/discriminator_training/loss/gradient_penalty_loss_fraction",
                                     gradient_penalty.item() * self.weight_gradient_penalty / loss.item())


            p_corr = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / confusion_matrix.sum()
            p_corr_learner = confusion_matrix[0, 0] / confusion_matrix[0].sum()
            p_corr_reference = confusion_matrix[1, 1] / confusion_matrix[1].sum()
            
            logger.store("imitation/discriminator_training/p_corr", p_corr)
            logger.store("imitation/discriminator_training/p_corr_learner", p_corr_learner)
            logger.store("imitation/discriminator_training/p_corr_reference", p_corr_reference)
            logger.store("imitation/discriminator_training/confusion_matrix", 
                         list(confusion_matrix.flatten()), 
                         raw=True,
                         print=False)

            it += 1
        if self.n_discriminator_updates % self.update_discriminator_every == 0:
            self.frozen_discriminator.load_state_dict(self.discriminator.state_dict())

    def discriminator_cost(self, observations, next_observations):
        concatenated = np.concatenate(
            [observations, next_observations], axis=1
        )
        with torch.no_grad():
            pred = self.frozen_discriminator(
                torch.tensor(concatenated, dtype=torch.float32).to(self.device)
            ).cpu().numpy().flatten()
        self.discriminator_running_mean_and_var = self.mean_and_var_update.reduce(
            pred, initial=self.discriminator_running_mean_and_var
        )
        if self.standarize_discriminator_output:
            cost = -(pred - self.discriminator_running_mean_and_var[0]) / np.sqrt(
            self.discriminator_running_mean_and_var[1]
        )
        else:
            cost = -pred
        cost = np.maximum(cost, 0)
        cost = np.clip(cost, -1, 1)
        cost *= self.imitation_cost_multiplier

        logger.store("imitation/cost/discriminator_output/p_identified",
                     (pred <= 0).sum() / len(pred))
        logger.store("imitation/cost/discriminator_output/mean", pred.mean())
        logger.store("imitation/cost/discriminator_output/std", pred.std())
        logger.store("imitation/cost/discriminator_output_running_vars/mean",
                     self.discriminator_running_mean_and_var[0])
        logger.store("imitation/cost/discriminator_output_running_vars/std",
                     np.sqrt(self.discriminator_running_mean_and_var[1]))
        logger.store("train/cost/discriminator_cost",
                     cost, stats=True)
        return cost

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
                     discriminator_cost / info["costs"] / 2, stats=True)
        logger.store("train/cost/environment_costs",
                     (info["costs"] - discriminator_cost / 2) * 2, stats=True)
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
        if self.test_mode == "mujoco_w_muscles":
            test_mujoco(env,
                        agent,
                        steps,
                        const_fn,
                        test_params,
                        data_path=data_path)

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
            discriminator_cost = self.discriminator_cost(
                self.agent.last_observations,
                observations
            )
            info["costs"] += discriminator_cost
            info["costs"] /= 2

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
                if self.all_trajectories:
                    self.train_discriminator(
                        self.agent.replay.get_keys_from_entire_histroy("observations", "next_observations")
                    )
                else:
                    self.train_discriminator(
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
