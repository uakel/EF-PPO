"""Script for hyper parameter search."""
import os
import traceback

import torch

# Agent building
from ef_ppo.ef_ppo import DEP_EF_PPO
from ef_ppo.hl_segment import HLSegment
from ef_ppo.models import HLActorCritic
from ef_ppo.hl_segment import HLSegment
from ef_ppo.critics import VRegression
from deprl.vendor.tonic.torch import models, normalizers, updaters

# Environment building
tonic_conf = "import sconegymdeprl"


# Training
from ef_ppo import custom_distributed
from deprl.utils import load_checkpoint, prepare_params

def make_agent():
    return DEP_EF_PPO(
      replay=HLSegment(
        size=128,
        discount_factor=0.99,
        trace_decay=0.9,
        batch_size=128,
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
        optimizer=lambda params: torch.optim.Adam(params, lr=3e-5),
        entropy_coeff=0.001,
        ratio_clip=0.07, 
      ), 
      h_critic_updater=VRegression(
        optimizer=lambda params: torch.optim.Adam(params, lr=4e-5),
      ),
      l_critic_updater=VRegression(
        optimizer=lambda params: torch.optim.Adam(params, lr=4e-5),
      ),
      min_budget=-2,
      max_budget=0,
    )

def make_trainer():
    return ef_ppo.trainers.ef_ppo_trainer.EFPPOTrainer(
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
):
    """
    Trains an agent on an environment.
    """
    # In case no env_args are passed via the config
    config["env_args"] = {}

    # Build the training environment.
    _environment = "TODO"
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
    agent: DEP_EF_PPO = make_agent()

    # Set DEP parameters
    if hasattr(agent, "expl"):
        agent.expl.set_params(DEP_PARAMS)

    # Initialize the logger to get paths
    config = {
        "working_dir": "/home/leo/thesis/HyFyDyEFPPOExperiments/logs",
        "tonic": {"name": "TODO"},
    }
    logger.initialize(
        script_path=__file__,
        config=config,
        test_env=test_environment,
        resume=True,
    )
    path = logger.get_path()

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
        full_save=True,
    )

    # Train.
    try:
        trainer.run(config, **time_dict) # type: ignore
    except Exception as e:
        logger.log(f"trainer failed. Exception: {e}")
        traceback.print_tb(e.__traceback__)

def set_tensor_device():
    # use CUDA or apple metal
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        logger.log("CUDA detected, storing default tensors on it.")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")
        logger.log("MPS detected, storing default tensors on it.")
    else:
        logger.log("No CUDA or MPS detected, running on CPU")


def main():
    set_tensor_device()
    train()


if __name__ == "__main__":
    main()
