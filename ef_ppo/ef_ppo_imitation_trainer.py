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
        replace_checkpoint=False,
        max_budget=None,
        data_path=lambda env: env.environments[0].sim.model,
        adaptive_budget=True,
        reference_dataset_path="dataset.npz",
        discriminator_optimizer=torch.optim.Adam,
        optimizer_kwargs={"lr" : 0.0005},
        weight_imitation=0.5,
        weight_gradient_penalty=5,
        imitation_cost_multiplier=0.005,
        device="cuda",
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
        self.weight_imitation = weight_imitation
        self.weight_gradient_penalty = weight_gradient_penalty
        self.imitation_cost_multiplier = imitation_cost_multiplier
        self.discriminator_running_mean_and_var = np.zeros(2)
        def mean_and_var_update(x, y):
            add = np.array([y, (y - x[0]) ** 2])
            return self.imitation_cost_alpha * x + (1 - self.imitation_cost_alpha) * add

        self.mean_and_var_update = np.frompyfunc(mean_and_var_update, 2, 1)

    def initialize(
        self, agent, environment, 
        test_environment=None, 
        full_save=False
    ):
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

    def train_discriminator(self, learner_dataset, epochs=1, batch_size=256):
        for _ in range(epochs):
            # We use the confusion matrix in the following way:
            # confusion_matrix[0, 0] = True positives, with positive being the leaner correcly identified as learner
            # confusion_matrix[0, 1] = False positives, 
            # confusion_matrix[1, 0] = False negatives, 
            # confusion_matrix[1, 1] = True negatives, with negative being the reference correctly identified as reference
            confusion_matrix = np.zeros((2, 2))
            for reference, learner in self.data_iterator(learner_dataset, batch_size):
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

                gradient_penalty = torch.mean(torch.norm(grad, dim=1) ** 2)

                pred_learner = self.discriminator(learner)
                loss = self.weight_imitation * F.binary_cross_entropy_with_logits(pred_learner, 
                                                                                  torch.zeros_like(pred_learner)) \
                     + self.weight_imitation * F.binary_cross_entropy_with_logits(pred_reference, 
                                                                                  torch.ones_like(pred_reference)) \
                     + self.weight_gradient_penalty * gradient_penalty
                # -> minimize pred_learner, maximize pred_reference
                loss.backward()
                self.discriminator_optimizer.step()
                logger.store("train/discriminator_loss", loss.item(), stats=True)

                confusion_matrix[0, 0] += (pred_learner < 0).sum().item()
                confusion_matrix[0, 1] += (pred_learner > 0).sum().item()
                confusion_matrix[1, 0] += (pred_reference < 0).sum().item()
                confusion_matrix[1, 1] += (pred_reference > 0).sum().item()

            logger.store("train/discriminator_confusion_matrix", confusion_matrix.flatten(), raw=True)

            acc = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / confusion_matrix.sum()
            sens = confusion_matrix[0, 0] / confusion_matrix[0].sum()
            spec = confusion_matrix[1, 1] / confusion_matrix[1].sum()
            F1 = 2 * sens * spec / (sens + spec)

            logger.store("train/discriminator_accuracy", acc, stats=True)
            logger.store("train/discriminator_sensitivity", sens, stats=True)
            logger.store("train/discriminator_specificity", spec, stats=True)
            logger.store("train/discriminator_F1", F1, stats=True)

    def discriminator_cost(self, observations, next_observations):
        concatenated = np.concatenate(
            [observations, next_observations], axis=1
        )
        with torch.no_grad():
            pred = self.discriminator(
                torch.tensor(concatenated, dtype=torch.float32).to(self.device)
            ).cpu().numpy().flatten()
        self.discriminator_running_mean_and_var = self.mean_and_var_update.reduce(
            pred, initial=self.discriminator_running_mean_and_var
        )
        logger.store("train/discriminator_mean", self.discriminator_running_mean_and_var[0])
        logger.store("train/discriminator_var", self.discriminator_running_mean_and_var[1])
        cost = -(pred - self.discriminator_running_mean_and_var[0]) / np.sqrt(
            self.discriminator_running_mean_and_var[1]
        )
        return cost

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

            # Update discriminator
            if (self.agent.replay.index + 1 == self.agent.replay.max_size or
                self.agent.replay.index * len(observations) + 1 
                    % self.reference_length == 0):
                self.train_discriminator(
                    self.agent.replay.get_full("observations", "next_observations")
                )

            # Calculate discriminator cost
            discriminator_cost = self.discriminator_cost(
                self.agent.last_observations,
                observations
            )

            # Store in logger
            logger.store("train/discriminator_cost",
                         discriminator_cost, stats=True)

            # Integrate discriminator cost
            info["costs"] += self.imitation_cost_multiplier * discriminator_cost
            info["costs"] /= 2

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
