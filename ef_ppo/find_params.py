"""Script for hyper parameter search."""
import os
import traceback

import torch
from .logger import initialize, get_path, log

# Agent building
from ef_ppo.ef_ppo import DEP_EF_PPO
from ef_ppo.hl_segment import HLSegment
from ef_ppo.models import HLActorCritic
from ef_ppo.hl_segment import HLSegment
from ef_ppo.critics import VRegression
from deprl.vendor.tonic.torch import models, normalizers, updaters

# Environment building
tonic_conf = {
    "header": "import sconegymdeprl",
    "parallel": 16,
    "sequential": 4,
}

# Cluster utils
from cluster_utils import cluster_main

# Training
from ef_ppo import custom_distributed
from deprl.utils import load_checkpoint, prepare_params
from ef_ppo.trainers import ef_ppo_trainer

def make_agent(
        lr_actor=3e-5,
        lr_critics=4e-5,
        clip_ratio=0.07,
        batch_size=128,
    ):
    return DEP_EF_PPO(
      replay=HLSegment(
        size=128,
        discount_factor=0.99,
        trace_decay=0.9,
        batch_size=batch_size,
        h_term_penalty=-1.2,
        l_term_penalty=-1.2,
      ), 
      model=HLActorCritic(
        actor=models.Actor(
          encoder=models.ObservationEncoder(),
          torso=models.MLP((256, 256), torch.nn.ReLU),
          head=models.GaussianPolicyHead(),
        ),
        l_critic=models.Critic(
          encoder=models.ObservationEncoder(),
          torso=models.MLP((256, 256), torch.nn.ReLU),
          head=models.ValueHead(),
        ), 
        h_critic=models.Critic(
          encoder=models.ObservationEncoder(),
          torso=models.MLP((256, 256), torch.nn.ReLU),
          head=models.ValueHead(),
        ), 
        observation_normalizer=normalizers.MeanStd(), 
      ), 
      actor_updater=updaters.actors.ClippedRatio(
        optimizer=lambda params: torch.optim.Adam(params, lr=lr_actor),
        entropy_coeff=0.001,
        ratio_clip=clip_ratio, 
      ), 
      h_critic_updater=VRegression(
        optimizer=lambda params: torch.optim.Adam(params, lr=lr_critics),
      ),
      l_critic_updater=VRegression(
        optimizer=lambda params: torch.optim.Adam(params, lr=lr_critics),
      ),
      min_budget=-2,
      max_budget=0,
    )

def make_trainer():
    return ef_ppo_trainer.EFPPOTrainer(
      steps=int(50e6),
      epoch_steps=int(16 * 4 * 128 * 50), 
      save_steps=int(16 * 4 * 128 * 50),
      test_episodes=100,
      discount=0.99,
      use_env_constraint=True,
      min_budget=-2,
      max_budget=0,
      test_fn="ef_ppo.test_fns.hyfydy:test",
      constraint_type="max",
      budget_update="modified",
    )

DEP_PARAMS = {
  "bias_rate": 0.002,
  "buffer_size": 200,
  "intervention_length": 8,
  "intervention_proba": 0.00371,
  "kappa": 1000,
  "normalization": "independent",
  "q_norm_selector": "l2",
  "regularization": 32,
  "s4avg": 2,
  "sensor_delay": 1,
  "tau": 40,
  "test_episode_every": 3,
  "time_dist": 5,
  "with_learning": True,
}

def train(
    sequential=4,
    batch_size=128,
    lr_actor=3e-5,
    lr_critics=4e-5,
    clip_ratio=0.07,
    **params,
):
    """
    Trains an agent on an environment.
    """
    # Prepare the parameters.
    tonic_conf["sequential"] = sequential

    # In case no env_args are passed via the config
    config = {}
    config["env_args"] = {}

    # Build the training environment.
    _environment = 'deprl.environments.Gym("walk_h0918-v42", max_episode_steps=1000, scaled_actions=False)'
    environment = custom_distributed.distribute(
        environment=_environment,
        tonic_conf=tonic_conf,
        env_args=None,
    )
    environment.initialize(seed=0)

    # Build the testing environment.
    _test_environment = _environment
    test_environment = custom_distributed.distribute(
        environment=_test_environment,
        tonic_conf=tonic_conf,
        env_args=None,
        parallel=1,
        sequential=1,
    )
    test_environment.initialize(seed=1)

    # Build the agent.
    agent: DEP_EF_PPO = make_agent(
        lr_actor=lr_actor,
        lr_critics=lr_critics,
        clip_ratio=clip_ratio,
        batch_size=batch_size,
    )
    agent.initialize(test_environment.environments[0].observation_space,
                     test_environment.environments[0].action_space,
                     seed=0)

    # Set DEP parameters
    if hasattr(agent, "expl"):
        agent.expl.set_params(DEP_PARAMS)

    # Initialize the logger to get paths
    str_sequential = str(sequential).replace(".", "_")
    str_lr_actor = str(lr_actor).replace(".", "_")
    str_lr_critics = str(lr_critics).replace(".", "_")
    str_clip_ratio = str(clip_ratio).replace(".", "_")
    str_batch_size = str(batch_size).replace(".", "_")
    config = {
        "working_dir": "/home/franzleo/thesis/param_search/",
        "tonic": {"name": f"ef_ppo_n{str_sequential}_lra{str_lr_actor}_lrc{str_lr_critics}_c{str_clip_ratio}_b{str_batch_size}"},
    }
    logger = initialize(
        script_path=__file__,
        config=config,
        test_env=test_environment,
        resume=True,
    )
    path = get_path()

    # Process the checkpoint path same way as in tonic_conf.play
    checkpoint_path = os.path.join(path, "checkpoints")

    time_dict = {"steps": 0, "epochs": 0, "episodes": 0}
    (
        _,
        checkpoint_path,
        loaded_time_dict,
    ) = load_checkpoint(checkpoint_path, checkpoint="last")
    time_dict = time_dict if loaded_time_dict is None else loaded_time_dict

    if checkpoint_path:
        # Load the logger from a checkpoint.
        logger.load(checkpoint_path, time_dict)
        # Load the weights of the agent form a checkpoint.
        agent.load(checkpoint_path)

    # Build the trainer.
    trainer = make_trainer()
    trainer.initialize(
        agent=agent,
        environment=environment,
        test_environment=test_environment,
        full_save=False,
    )

    # Train.
    score = -999
    score = trainer.run(config, **time_dict) # type: ignore
    print(f"Score: {score}")
    metrics = {"score": score}
    return metrics


def set_tensor_device():
    # use CUDA or apple metal
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
    else:
        torch.set_default_device("cpu")


def main(**params):
    set_tensor_device()
    score = train(**params)
    return score


if __name__ == "__main__":
    main()
