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
                 standarize_output: bool=False,
                 exponential_mean_discounting=0.9999,
                 imitation_cost_multiplier=1.0,
                 activation=torch.nn.ReLU,
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs={"lr": 1e-4},
                 weight_imitation=1.0,
                 weight_gradient_penalty=0,
                 gradient_steps=8,
                 update_frozen_every=1,
                 device="cpu",
                 ):
        # Reference dataset
        self.reference_dataset = reference_dataset
        self.reference_length = self.reference_dataset["observations"].shape[0]
        observation_dimension = self.reference_dataset["observations"].shape[1]

        # Network setup
        self.regressor = Regressor(
            2 * observation_dimension,
            hidden_dims,
            activation=activation
        ).to(device)
        self.frozen_regressor = Regressor(
            2 * observation_dimension,
            hidden_dims,
            activation=activation
        ).to(device)
        self.frozen_regressor.load_state_dict(self.regressor.state_dict())

        # Standartization of the discriminator output
        self.standarize_output: bool = standarize_output
        self.exponential_mean_discounting = exponential_mean_discounting
        self.output_running_mean_and_var = np.zeros(2)
        def mean_and_var_update(mean_and_var: np.ndarray,
                                y: float) -> np.ndarray:
            add = np.array([y, (y - mean_and_var[0]) ** 2])
            return self.exponential_mean_discounting * mean_and_var + (1 - self.exponential_mean_discounting) * add
        self.mean_and_var_update = np.frompyfunc(mean_and_var_update, 2, 1)

        # Cost parameters
        self.imitation_cost_multiplier = imitation_cost_multiplier

        # Training parameters
        self.optimizer = optimizer(
            self.regressor.parameters(),
            **optimizer_kwargs
        )
        self.weight_imitation = weight_imitation
        self.weight_gradient_penalty = weight_gradient_penalty
        self.gradient_steps = gradient_steps
        self.n_discriminator_updates = 0
        self.update_frozen_every = update_frozen_every

        self.device = device

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

    def update(self, learner_dataset, epochs=1, batch_size=128):
        self.n_discriminator_updates += 1
        for _ in range(epochs):
            confusion_matrix = np.zeros((2, 2))
            it = 0
            for reference, learner in self.data_iterator(learner_dataset, batch_size):
                if it >= self.gradient_steps:
                    break
                self.optimizer.zero_grad()

                reference.requires_grad = True
                pred_reference = self.regressor(reference)
                grad = torch.autograd.grad(
                    outputs=pred_reference,
                    inputs=reference,
                    grad_outputs=torch.ones_like(pred_reference),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

                gradient_penalty = torch.mean(torch.norm(grad, dim=1) ** 2)

                pred_learner = self.regressor(learner)

                loss = self.weight_imitation\
                     * F.binary_cross_entropy_with_logits(pred_learner, 
                                                          torch.zeros_like(pred_learner)) \
                     + self.weight_imitation\
                     * F.binary_cross_entropy_with_logits(pred_reference, 
                                                          torch.ones_like(pred_reference)) \
                     + self.weight_gradient_penalty * gradient_penalty
                # -> minimize pred_learner, maximize pred_reference
                loss.backward()
                self.optimizer.step()

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
        if self.n_discriminator_updates % self.update_frozen_every == 0:
            self.frozen_regressor.load_state_dict(self.regressor.state_dict())
    
    def cost(self, observations, next_observations):
        concatenated = np.concatenate(
            [observations, next_observations], axis=1
        )
        with torch.no_grad():
            pred  = self.frozen_regressor(
                torch.tensor(concatenated, dtype=torch.float32).to(self.device)
            ).cpu().numpy().flatten()
        if self.standarize_output:
            cost = -(pred - self.output_running_mean_and_var[0]) / np.sqrt(
            self.output_running_mean_and_var[1]
        )
        else:
            cost = -pred

        self.output_running_mean_and_var = self.mean_and_var_update.reduce(
            pred,
            initial=self.output_running_mean_and_var
        )
        cost = np.maximum(cost, 0) 
        cost = np.clip(cost, -1, 1)
        cost *= self.imitation_cost_multiplier

        logger.store("imitation/cost/discriminator_output/p_identified",
                     (pred <= 0).sum() / len(pred))
        logger.store("imitation/cost/discriminator_output/mean", pred.mean())
        logger.store("imitation/cost/discriminator_output/std", pred.std())
        logger.store("imitation/cost/discriminator_output_running_vars/mean",
                     self.output_running_mean_and_var[0])
        logger.store("imitation/cost/discriminator_output_running_vars/std",
                     np.sqrt(self.output_running_mean_and_var[1]))
        logger.store("train/cost/discriminator_cost",
                     cost, stats=True)
        return cost
