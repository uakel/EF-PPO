"""Script used to play with trained agents."""

import argparse
import os

import numpy as np

from deprl import env_wrappers, mujoco_render
from deprl.utils import load_checkpoint
from deprl.vendor.tonic import logger


def check_args(args):
    if args["path"] is None and args["checkpoint_file"] is None:
        raise Exception(
            "You need to specify either a <--path> or a <--checkpoint_file>"
        )

    if args["path"] is not None and args["checkpoint_file"] is not None:
        raise Exception(
            "Do not simultaneously specify <--checkpoint_file> and \
                        <--path>."
        )

    if args["checkpoint"] != "last" and args["checkpoint_file"] is not None:
        raise Exception(
            "Do not simultaneously specify a checkpoint step with <--checkpoint> and a checkpoint file with \
                        <--checkpoint_file>."
        )
    if args["checkpoint_file"] is not None:
        if ("checkpoints" not in args["checkpoint_file"]) or (
            ".pt" not in args["checkpoint_file"]
        ):
            raise Exception(
                f'Invalid <--checkpoint_file> given: {args["checkpoint_file"]}'
            )

    if args["path"] is not None:
        assert os.path.isfile(os.path.join(args["path"], "config.yaml"))

def get_paths(path, checkpoint, checkpoint_file):
    """
    Checkpoints can be given as number e.g. <--checkpoint 1000000> or as file paths
    e.g. <--checkpoint_file path/checkpoints/step_1000000.pt'>
    This function handles this functionality.
    """
    if checkpoint_file is not None:
        path = checkpoint_file.split("checkpoints")[0]
        checkpoint = checkpoint_file.split("step_")[1].split(".")[0]
    checkpoint_path = os.path.join(path, "checkpoints")
    return path, checkpoint, checkpoint_path

def play_gym(agent, environment, deterministic, budget, num_episodes, no_render):
    """Launches an agent in a Gym-based environment."""
    observations = environment.reset()
    muscle_states = environment.muscle_states

    score = 0
    length = 0
    min_reward = float("inf")
    max_reward = -float("inf")
    global_min_reward = float("inf")
    global_max_reward = -float("inf")
    steps = 0
    episodes = 0

    while True:
        if budget == "bisect":
            if not deterministic:
                actions, budget_star = agent.test_step(
                    observations, steps,
                )
            else:
                actions, budget_star = agent.deterministic_opt_step(
                    observations, muscle_states=muscle_states, steps=1e6
                )
        else:
            if not deterministic:
                actions, budget_star = agent.step(
                    np.atleast_2d(observations), steps, np.atleast_1d(float(budget)),
                    muscle_states=muscle_states,
                    greedy_episode=True,
                ), budget
            else:
                actions, budget_star = agent.deterministic_step(
                    observations, np.atleast_1d(float(budget)),
                ), budget
        if len(actions.shape) > 1:
            actions = actions[0, :]
        observations, reward, done, info = environment.step(actions)
        muscle_states = environment.muscle_states
        if not no_render:
            mujoco_render(environment)

        steps += 1
        score += reward
        min_reward = min(min_reward, reward)
        max_reward = max(max_reward, reward)
        global_min_reward = min(global_min_reward, reward)
        global_max_reward = max(global_max_reward, reward)
        length += 1

        if done or length >= environment.max_episode_steps:
            episodes += 1

            print()
            print(f"Episodes: {episodes:,}")
            print(f"Score: {score:,.3f}")
            print(f"Length: {length:,}")
            print(f"Terminal: {done:}")
            print(f"Min reward: {min_reward:,.3f}")
            print(f"Max reward: {max_reward:,.3f}")
            print(f"Global min reward: {min_reward:,.3f}")
            print(f"Global max reward: {max_reward:,.3f}")
            observations = environment.reset()
            muscle_states = environment.muscle_states

            score = 0
            length = 0
            min_reward = float("inf")
            max_reward = -float("inf")
            if episodes >= num_episodes:
                break

def play(
    path,
    checkpoint,
    seed,
    header,
    agent,
    environment,
    deterministic,
    budget,
    no_render,
    num_episodes,
    checkpoint_file,
):
    """Reloads an agent and an environment from a previous experiment."""

    logger.log(f"Loading experiment from {path}")
    # Load config file and checkpoint path from folder
    path, checkpoint, checkpoint_path = get_paths(
        path, checkpoint, checkpoint_file
    )
    config, checkpoint_path, _ = load_checkpoint(checkpoint_path, checkpoint)

    # Get important info from config
    assert config is not None
    header = header or config["tonic"]["header"]
    agent = agent or config["tonic"]["agent"]
    environment = environment or config["tonic"]["test_environment"]
    environment = environment or config["tonic"]["environment"]

    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header)

    # Build the agent.
    if not agent:
        raise ValueError("No agent specified.")
    agent = eval(agent)

    # Build the environment.
    environment = eval(environment)
    environment.seed(seed)
    environment = env_wrappers.apply_wrapper(environment)
    if config and "env_args" in config:
        environment.merge_args(config["env_args"])
        environment.apply_args()

    # Adapt mpo specific settings
    if config and "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    # Initialize the agent.
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )

    if checkpoint_path:
        agent.load(checkpoint_path, only_checkpoint=True)
    play_gym(agent, environment, deterministic, budget, num_episodes, no_render)


if __name__ == "__main__":
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--budget", type=str, default="bisect")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--header", default=None)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--checkpoint_file", default=None)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument("--environment", "--env")
    args = vars(parser.parse_args())
    check_args(args)
    play(**args)
