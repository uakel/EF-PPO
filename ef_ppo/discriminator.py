from typing import *
import numpy as np
import torch
import torch.nn.functional as F
from ef_ppo import logger

SHAPING_TYPE = Literal["neg_fancy", "fancy", "vanilla", "none"]
ELEMENTWISE_LOSS_TYPE = Literal["square", "bce", "shifted_l1", "oiw", "smooth_l1"]

class Regressor(torch.nn.Module):
    def __init__(
        self, 
        input_dimension: int, 
        hidden_dims: List[int], 
        activation: torch.nn.Module=torch.nn.ReLU
    ):  
        """
        MLP for regression
        """
        super().__init__()
        # Construct the dimensions
        dims = [input_dimension] + hidden_dims + [1]
        layers = []
        # Construct the layers
        for this_dim, next_dim in zip(dims[:-1], dims[1:]):
            layers.append(torch.nn.Linear(this_dim, next_dim))
            layers.append(activation())
        # Remove the last activation
        layers.pop()
        # Construct the model
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Discriminator():
    def __init__(
        self, 
        reference_dataset: np.ndarray, 
        hidden_dims: List[int],
        reward_shaping: SHAPING_TYPE = "none",
        constraint_shaping: SHAPING_TYPE = "none",
        mu_target: float = 0.0025,
        std_target: float = 0.0025,
        exponential_mean_discounting: float =0.9999,
        imitation_reward_weight: float =1.0,
        imitation_constraint_weight: float =1.0,
        imitation_constraint_allowance: float =0.2,
        activation: torch.nn.Module = torch.nn.ReLU,  
        elementwise_loss: ELEMENTWISE_LOSS_TYPE = "smooth_l1",
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: Dict ={"lr": 1e-4},
        batch_size: int = 32,
        weight_imitation: float = 1.0,
        weight_gradient_penalty: float = 0,
        gradient_steps: int | float = 8,
        update_frozen_every: int = 1,
        device: Literal["cuda", "cpu"] = "cpu",
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
        self.reward_shaping: SHAPING_TYPE = reward_shaping
        self.constraint_shaping: SHAPING_TYPE = constraint_shaping
        self.mu_target = mu_target
        self.std_target = std_target
        self.exponential_mean_discounting = exponential_mean_discounting
        self.output_running_mean_and_var = np.ones(2)
        def mean_and_var_update(mean_and_var: np.ndarray,
                                y: float) -> np.ndarray:
            add = np.array([y, (y - mean_and_var[0]) ** 2])
            return (self.exponential_mean_discounting * mean_and_var 
                    + (1 - self.exponential_mean_discounting) * add)
        self.mean_and_var_update = np.frompyfunc(mean_and_var_update, 2, 1)

        # Objective parameters
        self.imitation_reward_weight = imitation_reward_weight
        self.imitation_constraint_weight = imitation_constraint_weight
        self.imitation_constraint_allowance = imitation_constraint_allowance

        # Training parameters
        self.elementwise_loss = elementwise_loss
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

    def update(
        self,
        pi_D: Dict[str, np.ndarray],
        epochs=1
    ):
        """
        Update the discriminator with the policy data
        """
        log_pre = "imitation/discriminator_training/"
        self.n_discriminator_updates += 1
        for _ in range(epochs):
            conf_mat = np.zeros((2, 2))
            it = 0
            for ref, pi in self._data_iterator(pi_D, self.batch_size):
                if it >= self.gradient_steps:
                    break
                self.optimizer.zero_grad()

                # Compute gradient penalty
                grad_pen, p_ref = self._compute_gradient_penalty(ref)

                # Compute class. loss
                p_pi = self.regressor(pi)
                loss = self._compute_class_loss(p_ref, p_pi)
                loss += grad_pen * self.weight_gradient_penalty
                loss.backward()

                # Update
                self.optimizer.step()

                # Log
                self._log_training_predicitons(p_pi, p_ref, grad_pen, loss, log_pre)
                self._update_conf_mat(conf_mat, p_ref, p_pi)

                # Update iteration counter
                it += 1

            # Log metrics
            self._make_and_log_metrics_from_confusion_matrix(conf_mat, log_pre)

        # Update frozen regressor
        if self.n_discriminator_updates % self.update_frozen_every == 0:
            self.frozen_regressor.load_state_dict(self.regressor.state_dict())

    def predict(self, observations: np.ndarray, next_observations: np.ndarray) -> np.ndarray:
        """
        Lets the discriminator predict the logits
        """
        concatenated = np.concatenate(
            [observations, next_observations], axis=1
        )
        with torch.no_grad():
            pred = self.frozen_regressor(
                torch.tensor(concatenated, dtype=torch.float32).to(self.device)
            ).cpu().numpy().flatten()

        # Log and return
        self._log_policy_prediction_metrics(pred)
        return pred

    def update_mean_and_var(self, pred: np.ndarray):
        """
        Update the running mean and variance of the discriminator output
        """
        self.output_running_mean_and_var = self.mean_and_var_update.reduce(
            pred,
            initial=self.output_running_mean_and_var # type: ignore
        )

    def reward(self, pred: np.ndarray) -> np.ndarray:
        """
        Compute the policy reward from the discriminator predictions
        """
        reward = self._apply_shaping(pred.copy(), self.reward_shaping)
        reward *= self.imitation_reward_weight
        return reward

    def constraint(self, pred: np.ndarray) -> np.ndarray:
        """
        Compute the constraint evaluations from the discriminator predictions
        """
        constraint = -self._apply_shaping(pred.copy(), self.constraint_shaping)
        constraint *= self.imitation_constraint_weight
        return constraint - self.imitation_constraint_allowance

    # Reward and constraint shaping
    def _apply_shaping(self, pred: np.ndarray, shaping: SHAPING_TYPE) -> np.ndarray:
        """
        Applies some shaping operation to the discriminator outputs
        """
        if shaping == "fancy":
            return self._fancy_std(pred)
        if shaping == "neg_fancy":
            return self._neg_fancy_std(pred)
        elif shaping == "vanilla":
            return self._vanilla_std(pred)
        elif shaping == "none":
            return pred
        else:
            raise ValueError("shaping must be one of 'fancy', 'neg_fancy', 'vanilla', 'none'")

    def _fancy_std(self, pred: np.ndarray) -> np.ndarray:
        """
        Fancy reward standardization
        """
        return (
            self.std_target * (pred - self.output_running_mean_and_var[0]) / 
                np.sqrt(self.output_running_mean_and_var[1]) 
            + np.minimum(
                self.mu_target, 
                -self.std_target * self.output_running_mean_and_var[0] / 
                np.sqrt(self.output_running_mean_and_var[1])
            )
        )
       
                            
    def _neg_fancy_std(self, pred: np.ndarray) -> np.ndarray:
        """
        Fancy reward standardization with r <= 0
        """
        return -np.maximum(
            -self.std_target * (pred - self.output_running_mean_and_var[0]) / 
            np.sqrt(self.output_running_mean_and_var[1]) 
            + np.minimum(
                self.mu_target, 
                -self.std_target * self.output_running_mean_and_var[0] / 
                np.sqrt(self.output_running_mean_and_var[1])
            ),
            0
        )

    def _vanilla_std(self, pred: np.ndarray) -> np.ndarray:
        """
        Vanilla reward standardization
        """
        return (pred - self.output_running_mean_and_var[0]) / np.sqrt(
            self.output_running_mean_and_var[1]
        )

    # Training helper functions
    def _data_iterator(
        self,
        pi_D: Dict[str, np.ndarray],
        batch_size: int=256
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Generator that yields batches made from the 
        reference and learner datasets
        """
        # Filter out the positive budgets
        postive_budgets = pi_D["budgets"] > 0
        logger.store("imitation/positive_budget_fraction",
                     sum(postive_budgets) / len(postive_budgets))
        
        shortest = min(self.reference_length, 
                       len(pi_D["observations"][postive_budgets]))
        ref_I = np.random.choice(
            len(self.ref_D["observations"]), shortest, replace=False
        )
        pi_I = np.random.choice(
            len(pi_D["observations"][postive_budgets]), shortest, replace=False
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
                    pi_D["observations"][postive_budgets][pi_I[i:i+batch_size]],
                    pi_D["next_observations"][postive_budgets][pi_I[i:i+batch_size]]
                ],
                axis=1
            )
            yield (torch.tensor(ref, dtype=torch.float32).to(self.device),
                   torch.tensor(pi, dtype=torch.float32).to(self.device))

    def _square_loss(
        self,
        p_ref: torch.Tensor,
        p_pi: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the square loss
        """
        return F.mse_loss(p_ref, torch.ones_like(p_ref)) + F.mse_loss(p_pi, -torch.ones_like(p_pi))

    def _bce_loss(
        self,
        p_ref: torch.Tensor,
        p_pi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross entropy loss
        """
        return F.binary_cross_entropy_with_logits(
            p_ref, torch.ones_like(p_ref)
        ) + F.binary_cross_entropy_with_logits(
            p_pi, torch.zeros_like(p_pi)
        )

    def _shifted_l1_loss(
        self,
        p_ref: torch.Tensor,
        p_pi: torch.Tensor
    ) -> torch.Tensor:
        """
        Shifted L1 loss
        """
        return torch.mean(
            torch.clamp(torch.abs(p_ref - 1) - 0.2, 0)
          + torch.clamp(torch.abs(p_pi + 1) - 0.2, 0)
        )

    def _oiw_loss(
        self,
        p_ref: torch.Tensor,
        p_pi: torch.Tensor
    ) -> torch.Tensor:
        """
        Only if wrong loss
        """
        return torch.mean(torch.clamp(p_ref, 0) + torch.clamp(-p_pi, 0))

    def smooth_l1_loss(
        self,
        p_ref: torch.Tensor,
        p_pi: torch.Tensor
    ) -> torch.Tensor:
        """
        Smooth L1 loss
        """
        return (
            F.smooth_l1_loss(p_ref, torch.ones_like(p_ref)) 
          + F.smooth_l1_loss(p_pi, -torch.ones_like(p_pi))
        )

    def _compute_class_loss(
        self,
        p_ref: torch.Tensor,
        p_pi: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the classificaiton loss
        """
        if self.elementwise_loss == "square":
            loss = self._square_loss(p_ref, p_pi)
        elif self.elementwise_loss == "bce":
            loss = self._bce_loss(p_ref, p_pi)
        elif self.elementwise_loss == "shifted_l1":
            loss = self._shifted_l1_loss(p_ref, p_pi)
        elif self.elementwise_loss == "oiw":
            loss = self._oiw_loss(p_ref, p_pi)
        elif self.elementwise_loss == "smooth_l1":
            loss = self.smooth_l1_loss(p_ref, p_pi)
        else:
            raise ValueError("elementwise_loss must be one of 'square', 'bce', 'shifted_l1', 'oiw'", "smooth_l1")
        # => maximize reference predictions
        # => minimize policy predictions
        return self.weight_imitation * loss

    def _compute_gradient_penalty(
        self, 
        ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the gradient penalty. This yields also the 
        predictions for the reference.
        """
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
        return grad_pen, p_ref

    # Logging helper functions
    def _log_training_predicitons(
        self, 
        p_pi: torch.Tensor, 
        p_ref: torch.Tensor, 
        grad_pen: torch.Tensor,
        loss: torch.Tensor,
        log_pre: str=""
    ):
        """
        Log several metrics related to the predictions
        """
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

    def _update_conf_mat(
        self, 
        conf_mat: np.ndarray,
        p_ref: torch.Tensor,
        p_pi: torch.Tensor
    ):
        """
        Updates a confusion matrix
        """
        with torch.no_grad():
            conf_mat[0, 0] += (p_pi <= 0).sum().item()
            conf_mat[0, 1] += (p_pi >= 0).sum().item()
            conf_mat[1, 0] += (p_ref < 0).sum().item()
            conf_mat[1, 1] += (p_ref > 0).sum().item()

    def _make_and_log_metrics_from_confusion_matrix(
        self,
        conf_mat: np.ndarray,
        log_pre: str=""
    ):
        """
        Make and log metrics from the confusion matrix
        """
        p_corr = (conf_mat[0, 0] + conf_mat[1, 1]) / conf_mat.sum()
        p_corr_learner = conf_mat[0, 0] / conf_mat[0].sum()
        p_corr_reference = conf_mat[1, 1] / conf_mat[1].sum()
        
        logger.store(log_pre + "p_corr", p_corr)
        logger.store(log_pre + "p_corr_learner", p_corr_learner)
        logger.store(log_pre + "p_corr_reference", p_corr_reference)
        logger.store(log_pre + "confusion_matrix", 
                     list(conf_mat.flatten()), 
                     stat_level="r")

    def _log_policy_prediction_metrics(
        self,
        pred: np.ndarray, 
        log_pre:str ="imitation/reward/"
    ):
        logger.store(log_pre + "discriminator_output/p_identified",
                     (pred <= 0).sum() / len(pred))
        logger.store(log_pre + "discriminator_output", pred, stat_level="ms")
        logger.store(log_pre + "discriminator_output_running_vars/mean",
                     self.output_running_mean_and_var[0])
        logger.store(log_pre + "discriminator_output_running_vars/std",
                     np.sqrt(self.output_running_mean_and_var[1]))
