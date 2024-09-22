import torch

from deprl.vendor.tonic.torch import models

class VRegression:
    """
    Value Regression with specifiable critic
    """
    def __init__(self, loss=None, optimizer=None, gradient_clip=0):
        self.loss = loss or torch.nn.MSELoss()
        self.optimizer = optimizer or (
            lambda params: torch.optim.Adam(params, lr=1e-3)
        )
        self.gradient_clip = gradient_clip

    def initialize(self, critic):
        self.critic = critic 
        self.variables = models.trainable_variables(self.critic) # Here the critic is directly specified
        self.optimizer = self.optimizer(self.variables)

    def __call__(self, observations, returns):
        self.optimizer.zero_grad()
        values = self.critic(observations)
        loss = self.loss(values, returns)
        loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.variables, self.gradient_clip)
        self.optimizer.step()

        return dict(loss=loss.detach(), v=values.detach())
