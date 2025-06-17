import torch
from ef_ppo.agents.ef_ppo import DEP_EF_PPO
from ef_ppo import logger
from ef_ppo.distributional_z_regressors.quantile_network import QuantileRegression

# Main class    
class Z_aware_EF_PPO(DEP_EF_PPO):
    """
    Z aware Epigraph Form PPO (EF-PPO).

    EF-PPO: https://arxiv.org/pdf/2305.14154.pdf
    """
    def __init__(self, model=None, replay=None, actor_updater=None,
                 h_critic_updater=None, l_critic_updater=None, z_regressor_updater=None,
                 log=True, budget_normalizer=1.0, min_budget=-2.7, max_budget=2.7):
        self.l_critic_updater = z_regressor_updater or\
                                QuantileRegression()
        super().__init__(
            model=model,
            replay=replay,
            actor_updater=actor_updater,
            h_critic_updater=h_critic_updater,
            l_critic_updater=l_critic_updater,
            log=log,
            budget_normalizer=budget_normalizer,
            min_budget=min_budget,
            max_budget=max_budget
        )

    def learned_z_update(self, observations, budget_stars):
        self.replay.store(
            observations=observations,
            budgets=budget_stars,
        )
        if self.replay.ready():
            self._learned_z_update()

    def _learned_z_update(self):
        for batch in self.replay.get_full(
            "observations",
            "budgets"
        ):
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}
            info = self.model.z_regressor(**batch)
            logger.store("z_regressor/loss", info["loss"])

    def learned_z_step(self, observations, steps, muscle_states=None):
        """
        Step method that returns the actions and log-probs
        """
        # Cast to tensor
        observations = torch.as_tensor(observations, dtype=torch.float32)
        distributions = self.model.z_regressor(observations)
        budget = distributions.mean()

        # Augment observation
        obs_and_budget = self.budget_augmented_obs(observations, budget)
        obs_and_budget = torch.atleast_2d(obs_and_budget)

        # Evaluate actor and sample action
        with torch.no_grad():
            distributions = self.model.actor(obs_and_budget)
            actions = distributions.mean()

        actions = actions.numpy(force=True)
        return actions
