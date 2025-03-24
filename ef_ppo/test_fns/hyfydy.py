from typing import *
import numpy as np

from ef_ppo import logger
from ef_ppo.ef_ppo import EF_PPO
from ef_ppo.custom_distributed import Parallel, Sequential

def R_h_update(
    R_h: float,
    h: float,
    gamma: float,
    l: int,
    aleph: float
):
    """
    Updates the constraint return.
    """
    R_h = np.maximum(
        R_h, 
        (1 - gamma) * aleph 
        + gamma ** l * h
    )
    aleph += gamma ** l * h
    return R_h, aleph

def R_r_update(
    R_r: float,
    r: float,
    gamma: float,
    l: int
):
    """
    Updates the return.
    """
    R_r = gamma ** l * R_r + r
    return R_r

def log(
    **kwargs
):
    """
    Logs the values.
    """
    for k, v in kwargs.items():
        logger.store("test/" + k, v, stat_level="ms")

def test(
    env: Union[Parallel, Sequential],
    agent: EF_PPO, 
    steps, 
    constraint_function, 
    test_episodes=10, 
    data_path=lambda env: env.environments[0].unwrapped.sim.data
):
    """
    Tests the EF-PPO agent on the test environment.
    """
    if not hasattr(agent, "started"):
        obs, _ = env.start()
        agent.started = True

    naked_env = env.environments[0] # type: ignore
    gamma = agent.replay.discount_factor

    all_speeds = []
    all_efforts = []
    all_ep_lengths = []

    for ep in range(test_episodes):
        R_r = 0
        R_h = -np.inf
        aleph = 0    
        length = 0
        if ep == 0:
            obs = naked_env.store_next_episode()[None, :]
        else:
            obs = naked_env.reset()[None, :] # type: ignore
        for i in range(1000):
            actions = agent.deterministic_opt_step(obs, steps) # type: ignore
            budget_star = agent.budget_star.copy()

            obs, _, info = env.step(actions)

            const_fn_evals = float(info["constraint"].copy()) 
            r = info["rewards"][0]
            effort = np.mean(naked_env.muscle_activity() ** 2)
            speed = naked_env.hyfydy_env.model_velocity()

            all_speeds.append(speed)
            all_efforts.append(effort)

            length += 1
            R_h, aleph = R_h_update(R_h, const_fn_evals, gamma, length, aleph)
            R_r = R_r_update(R_r, r, gamma, length)

            log(
                rewards=r,
                constraint_fn_evals=const_fn_evals,
                budget_star=budget_star,
                effort=effort,
                speed=speed
            )

            if i == 997:
                naked_env.write_now()
            if info["resets"][0]:
                log(
                    episode_length=length,
                    episode_return=R_r,
                    constraint_return=R_h
                )
                all_ep_lengths.append(length)
                break
    score = np.mean(1 - np.mean(all_efforts) / 0.1) * 100
    score *= np.mean(all_ep_lengths) / 100
    if np.mean(all_speeds) < 0.9:
        score = 0
    if np.mean(all_ep_lengths) < 120:
        score = 0
    return score
