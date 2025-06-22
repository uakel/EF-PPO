import numpy as np
import torch

class HLActorCritic(torch.nn.Module):
    def __init__(
        self,
        actor,
        h_critic,
        l_critic,
        observation_normalizer=None,
        return_normalizer=None,
    ):
        super().__init__()
        self.actor = actor
        self.h_critic = h_critic
        self.l_critic = l_critic
        self.observation_normalizer = observation_normalizer
        self.return_normalizer = return_normalizer

    def initialize(self, observation_space, action_space):
        # Append the budget dimension to the observation space
        low = np.append(observation_space.low, 0)
        high = np.append(observation_space.high, np.inf)
        observation_space = type(observation_space)(low=low, high=high) 

        if self.observation_normalizer:
            self.observation_normalizer.initialize(observation_space.shape)
        self.actor.initialize(
            observation_space, action_space, self.observation_normalizer
        )
        self.h_critic.initialize(
            observation_space,
            action_space,
            self.observation_normalizer,
            self.return_normalizer,
        )
        self.l_critic.initialize(
            observation_space,
            action_space,
            self.observation_normalizer,
            self.return_normalizer,
        )

class Distributional_z_HL_critic(HLActorCritic):
    def __init__(
        self,
        actor,
        h_critic,
        l_critic,
        z_regressor,
        observation_normalizer=None,
        return_normalizer=None,
    ):
        super().__init__(
            actor=actor,
            h_critic=h_critic,
            l_critic=l_critic,
            observation_normalizer=observation_normalizer,
            return_normalizer=return_normalizer,
        )
        self.z_regressor = z_regressor

    def initialize(self, observation_space, action_space):
        self.z_regressor.initialize(observation_space, action_space)
        super().initialize(observation_space, action_space)

class FourierObservationEncoder(torch.nn.Module):
    def __init__(
        self,
        observation_size,
        mapping_size=256,
        scale=10.0,
    ):
        self.observation_size = observation_size
        self.mapping_size = mapping_size
        self.shifts = torch.randn(mapping_size, observation_size) * scale

    def initialize(
        self,
        observation_space,
        action_space=None,
        observation_normalizer=None,
    ):
        self.observation_normalizer = observation_normalizer
        return self.observation_size

    def forward(self, observations):
        if self.observation_normalizer:
            observations = self.observation_normalizer(observations)
        shape = observations.shape
        observations = torch.cat([torch.cos(
            self.shifts @ observations.unsqueeze(-1)
        ),
        torch.sin(self.shifts @ observations.unsqueeze(-1))], dim=-1)
        return observations

