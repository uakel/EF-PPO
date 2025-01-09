from typing import *
import numpy as np
import torch
import torch.nn.functional as F
from ef_ppo import logger

class Regressor(torch.nn.Module):
    def __init__(self, 
                 input_dimension, 
                 hidden_dims, 
                 activation=torch.nn.ReLU):
        super().__init__()
        dims = [input_dimension] + hidden_dims + [1]
        layers = []
        for this_dim, next_dim in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(this_dim, next_dim))
            layers.append(activation())
        layers.pop()
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

class Discriminator():
    def __init__(self, 
                 reference_dataset, 
                 hidden_dims,
                 standardize_output : bool | Literal["fancy"] =False,
                 exponential_mean_discounting : float =0.9999,
                 imitation_reward_weight : float =1.0,
                 activation : torch.nn.Module = torch.nn.ReLU, # type: ignore
                 optimizer : torch.optim.Optimizer = torch.optim.Adam, # type: ignore
                 optimizer_kwargs : Dict ={"lr": 1e-4},
                 batch_size : int = 32,
                 weight_imitation : float = 1.0,
                 weight_gradient_penalty : float = 0,
                 gradient_steps : int | float = 8,
                 update_frozen_every : int = 1,
                 device : Literal["cuda", "cpu"] = "cpu",
                 ):
        # Reference dataset
        self.ref_D = reference_dataset
        self.reference_length = self.ref_D["observations"].shape[0]
        observation_dimension = self.ref_D["observations"].shape[1]

        # Network setup
        self.regressor = Regressor(
            2 * observation_dimension,
            hidden_dims,
            activation=activation # type: ignore
        ).to(device)
        self.frozen_regressor = Regressor(
            2 * observation_dimension,
            hidden_dims,
            activation=activation # type: ignore
        ).to(device)
        self.frozen_regressor.load_state_dict(self.regressor.state_dict())

        # Standartization of the discriminator output
        self.standardize_output = standardize_output
        self.exponential_mean_discounting = exponential_mean_discounting
        self.output_running_mean_and_var = np.ones(2)
        def mean_and_var_update(mean_and_var: np.ndarray,
                                y: float) -> np.ndarray:
            add = np.array([y, (y - mean_and_var[0]) ** 2])
            return (self.exponential_mean_discounting * mean_and_var 
                    + (1 - self.exponential_mean_discounting) * add)
        self.mean_and_var_update = np.frompyfunc(mean_and_var_update, 2, 1)

        # Cost parameters
        self.imitation_cost_multiplier = imitation_reward_weight

        # Training parameters
        self.optimizer = optimizer( # type: ignore
            self.regressor.parameters(),
            **optimizer_kwargs
        )
        self.batch_size = batch_size
        self.weight_imitation = weight_imitation
        self.weight_gradient_penalty = weight_gradient_penalty
        self.gradient_steps = gradient_steps
        self.n_discriminator_updates = 0
        self.update_frozen_every = update_frozen_every

        self.device = device

    def data_iterator(self, pi_D, batch_size=256):
        """
        Generator that yields batches made from the reference and learner dataset
        """
        shortest = min(self.reference_length, 
                       len(pi_D["observations"]))
        ref_I = np.random.choice(
            len(self.ref_D["observations"]), shortest, replace=False
        )
        pi_I = np.random.choice(
            len(pi_D["observations"]), shortest, replace=False
        )
        for i in range(0, shortest, batch_size):
            ref = np.concatenate(
                [
                    self.ref_D["observations"][ref_I[i:i+batch_size]],
                    self.ref_D["next_observations"][ref_I[i:i+batch_size]]
                ],
                axis=1
            )
            pi = np.concatenate(
                [
                    pi_D["observations"][pi_I[i:i+batch_size]],
                    pi_D["next_observations"][pi_I[i:i+batch_size]]
                ],
                axis=1
            )
            yield (torch.tensor(ref, dtype=torch.float32).to(self.device),
                   torch.tensor(pi, dtype=torch.float32).to(self.device))

    def _log_predicitons(self, p_pi, p_ref, grad_pen, loss, log_pre=""):
        with torch.no_grad():
            logger.store(log_pre + "pred_learner/mean", 
                         p_pi.mean().item())
            logger.store(log_pre + "pred_learner/std", 
                         p_pi.std().item())
            logger.store(log_pre + "pred_reference/mean",
                         p_ref.mean().item())
            logger.store(log_pre + "pred_reference/std",
                         p_ref.std().item())
            logger.store(log_pre + "loss/total", 
                         loss.item())
            logger.store(log_pre + "loss/gradient_penalty",
                         grad_pen.item() * self.weight_gradient_penalty)
            logger.store(log_pre + "loss/gradient_penalty_loss_fraction",
                         grad_pen.item() * self.weight_gradient_penalty / loss.item())

    def _make_and_log_metrics_from_confusion_matrix(self, conf_mat, log_pre=""):
        p_corr = (conf_mat[0, 0] + conf_mat[1, 1]) / conf_mat.sum()
        p_corr_learner = conf_mat[0, 0] / conf_mat[0].sum()
        p_corr_reference = conf_mat[1, 1] / conf_mat[1].sum()
        
        logger.store(log_pre + "p_corr", p_corr)
        logger.store(log_pre + "p_corr_learner", p_corr_learner)
        logger.store(log_pre + "p_corr_reference", p_corr_reference)
        logger.store(log_pre + "confusion_matrix", 
                     list(conf_mat.flatten()), 
                     raw=True,
                     print=False)

    def update(self, pi_D, epochs=1):
        """
        Update the discriminator
        """
        log_pre = "imitation/discriminator_training/"
        self.n_discriminator_updates += 1
        for _ in range(epochs):
            conf_mat = np.zeros((2, 2))
            it = 0
            for ref, pi in self.data_iterator(pi_D, self.batch_size):
                if it >= self.gradient_steps:
                    break
                self.optimizer.zero_grad()

                # Compute gradient penalty
                ref.requires_grad = True
                p_ref = self.regressor(ref)
                grad = torch.autograd.grad(
                    outputs=p_ref,
                    inputs=ref,
                    grad_outputs=torch.ones_like(p_ref),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                grad_pen = torch.mean(torch.norm(grad, dim=1) ** 2)

                # Compute class. loss
                p_pi = self.regressor(pi)
                loss = self.weight_imitation\
                     * F.binary_cross_entropy_with_logits(
                        p_pi, 
                        torch.zeros_like(p_pi)
                     ) + self.weight_imitation\
                     * F.binary_cross_entropy_with_logits(
                        p_ref, 
                        torch.ones_like(p_ref)
                     ) + self.weight_gradient_penalty * grad_pen
                # => minimize policy predictions
                # => maximize reference predictions

                # Update
                loss.backward()
                self.optimizer.step()

                # Log
                self._log_predicitons(p_pi, p_ref, grad_pen, loss, log_pre)
                with torch.no_grad():
                    conf_mat[0, 0] += (p_pi <= 0).sum().item()
                    conf_mat[0, 1] += (p_pi >= 0).sum().item()
                    conf_mat[1, 0] += (p_ref < 0).sum().item()
                    conf_mat[1, 1] += (p_ref > 0).sum().item()

                
                # Update iteration counter
                it += 1

            # Log metrics
            self._make_and_log_metrics_from_confusion_matrix(conf_mat, log_pre)

        # Update frozen regressor
        if self.n_discriminator_updates % self.update_frozen_every == 0:
            self.frozen_regressor.load_state_dict(self.regressor.state_dict())
    
    def reward(self, observations, next_observations):
        """
        Compute the reward for the discriminator
        """
        concatenated = np.concatenate(
            [observations, next_observations], axis=1
        )
        with torch.no_grad():
            pred  = self.frozen_regressor(
                torch.tensor(concatenated, dtype=torch.float32).to(self.device)
            ).cpu().numpy().flatten()
        if type(self.standardize_output) == str and self.standardize_output == "fancy":
            reward = -np.maximum(
                -0.0025 * (pred - self.output_running_mean_and_var[0]) / 
                    np.sqrt(
                        self.output_running_mean_and_var[1] 
                    ) + np.minimum(0.0025, -0.0025 * self.output_running_mean_and_var[0] / np.sqrt(
                        self.output_running_mean_and_var[1] 
                    )),
                0
            )
        elif type(self.standardize_output) == bool and self.standardize_output:
            reward = (pred - self.output_running_mean_and_var[0]) / np.sqrt(
            self.output_running_mean_and_var[1]
        )
        else:
            reward = pred

        self.output_running_mean_and_var = self.mean_and_var_update.reduce(
            pred,
            initial=self.output_running_mean_and_var
        )
        cost = np.maximum(cost, 0) 
        cost *= self.imitation_cost_multiplier

        logger.store("imitation/cost/discriminator_output/p_identified",
                     (pred <= 0).sum() / len(pred))
        logger.store("imitation/cost/discriminator_output", pred, stats=True)
        logger.store("imitation/cost/discriminator_output_running_vars/mean",
                     self.output_running_mean_and_var[0])
        logger.store("imitation/cost/discriminator_output_running_vars/std",
                     np.sqrt(self.output_running_mean_and_var[1]))
        logger.store("train/cost/discriminator_cost",
                     cost, stats=True)
        return cost
