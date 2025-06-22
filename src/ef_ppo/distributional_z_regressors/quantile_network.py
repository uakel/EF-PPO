import torch
import torch.nn as nn
import torch.nn.functional as F

from deprl.vendor.tonic.torch import models

class QuantileDistribution:
    def __init__(self, quantiles):
        self.quantiles = quantiles
        self.n_quantiles = quantiles.shape[-1]

    def sample(self):
        indices = torch.randint(0, self.n_quantiles, self.quantiles.shape[:-1])
        return torch.gather(self.quantiles, -1, indices[..., None])[..., 0]

    def mean(self):
        return self.quantiles.mean(dim=-1)

    def median(self):
        i = self.n_quantiles // 2
        return self.quantiles[..., i]

    def max_density(self):
        diffs = self.quantiles[...,:-1] - self.quantiles[...,1:]
        arg_min = diffs.argmin(dim=-1)
        return torch.gather(self.quantiles, -1, arg_min[..., None])
        
class QuantileRegressionHead(nn.Module):
    def __init__(self, num_quantiles=10):
        super().__init__()
        self.num_quantiles = num_quantiles

    def initialize(self, input_size, return_normalizer):
        self.offsets = nn.Linear(input_size, self.num_quantiles)
        self.base = nn.Linear(input_size, 1)

    def forward(self, inputs):
        b = self.base(inputs)                          
        o = F.softplus(self.offsets(inputs))           
        quantiles = torch.cumsum(o, dim=-1) + b 
        return QuantileDistribution(quantiles)

class QuantileRegression:
    def __init__(self, loss=None, optimizer=None):
        self.loss = loss or nn.HuberLoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3)
        )

    def initialize(self, model):
        self.model = model 
        num_quantiles = self.model.head.num_quantiles
        self.quantiles = torch.linspace(0, 1, num_quantiles)
        self.variables = models.trainable_variables(self.model)
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, budgets):
        self.optimizer.zero_grad()
        predicted_quantiles = self.model(observations).quantiles
        losses = self.loss(predicted_quantiles, budgets[..., None])
        losses = torch.where(
                predicted_quantiles > self.quantiles, 
                self.quantiles * losses,
                (1 - self.quantiles) * losses
        )
        loss = losses.mean()
        loss.backward()
        self.optimizer.step()

        return dict(loss=loss.detach().cpu().item(), v=predicted_quantiles.detach())

