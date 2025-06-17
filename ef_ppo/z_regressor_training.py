from typing import *
import numpy as np

from ef_ppo import logger
from ef_ppo.agents.z_aware_ef_ppo import Z_aware_EF_PPO
from ef_ppo.custom_distributed import Parallel, Sequential

N_FILLINGS = 5

def test(
    environment: Union[Parallel, Sequential],
    agent: Z_aware_EF_PPO, 
):
    """
    Tests the EF-PPO agent on the test environment.
    """
    n_iters = agent.replay.max_size * N_FILLINGS

    score = "TODO"
    observations = agent.last_observations
    for it in range(n_iters):
        actions = agent.deterministic_opt_step(observations, it, None)
        budget_stars = agent.budget_star.copy()
        observations, muscle_states, info = environment.step(actions)
        agent.learned_z_update(observations, budget_stars)

    return score
