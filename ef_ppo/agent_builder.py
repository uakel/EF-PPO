import torch
from ef_ppo.agents.z_aware_ef_ppo import Z_aware_EF_PPO
from ef_ppo.buffers.instantaneous_constraint_buffer import InstantaneousConstraintSegmentBuffer
from ef_ppo.models import Distributional_z_HL_critic
from deprl.vendor.tonic.torch import models, normalizers, updaters
from ef_ppo.distributional_z_regressors.quantile_network import QuantileRegressionHead, QuantileRegression
from ef_ppo.critics import VRegression

def z_agent_builder():
    # replay
    replay = InstantaneousConstraintSegmentBuffer(
        size=256,
        discount_factor=0.99,
        trace_decay=0.9,
        batch_size=128,
        h_term_penalty=-1,
        l_term_penalty=-1,
    )

    # model
    std_mlp_torso_maker = lambda: models.MLP((256, 256), torch.nn.ReLU)
    actor = models.Actor(
        encoder=models.ObservationEncoder(),
        torso=std_mlp_torso_maker(),
        head=models.GaussianPolicyHead()
    )
    critic_builder = lambda head: models.Critic(
        encoder=models.ObservationEncoder(),
            torso=std_mlp_torso_maker(),
            head=head()
        )
    model = Distributional_z_HL_critic(
        actor=actor,
        l_critic=critic_builder(models.ValueHead),
        h_critic=critic_builder(models.ValueHead),
        z_regressor=critic_builder(QuantileRegressionHead),
        observation_normalizer=normalizers.MeanStd()
    )

    # updaters
    actor_updater = updaters.ClippedRatio(
        optimizer=lambda params: torch.optim.Adam(params, lr=3e-5),
        entropy_coeff=0.001,
        ratio_clip=0.20 
    )
    h_critic_updater = VRegression(
        optimizer=lambda params: torch.optim.Adam(params, lr=4e-5),
    )
    l_critic_updater = VRegression(
        optimizer=lambda params: torch.optim.Adam(params, lr=4e-5),
    )
    z_regressor_updater = QuantileRegression(
        optimizer=lambda params: torch.optim.Adam(params, lr=4e-5),
    )
    agent = Z_aware_EF_PPO(
        replay=replay,
        model=model,
        actor_updater=actor_updater,
        h_critic_updater=h_critic_updater,
        l_critic_updater=l_critic_updater,
        z_regressor_updater=z_regressor_updater,
        max_budget=1,
    )
    return agent
