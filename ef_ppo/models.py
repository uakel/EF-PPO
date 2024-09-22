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
