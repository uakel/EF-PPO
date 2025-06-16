from .ef_ppo import EF_PPO

# Main class    
class Z_aware_EF_PPO(EF_PPO):
    """
    Z aware Epigraph Form PPO (EF-PPO).

    EF-PPO: https://arxiv.org/pdf/2305.14154.pdf
    """
    def __init__(
        self,
        model=None,
        replay=None,
        actor_updater=None,
        h_critic_updater=None,
        l_critic_updater=None, 
        log=True,
        budget_normalizer=1.0,
        min_budget=-2.7,
        max_budget=2.7,
    ):

        # Call the parent class __init__ function
        super().__init__(
            model=model,
            replay=replay,
            actor_updater=actor_updater,
            h_critic_updater=h_critic_updater,
            l_critic_updater=l_critic_updater,
            log=log,
            budget_normalizer=budget_normalizer,
            min_budget=min_budget,
            max_budget=max_budget,
        )

    def learned_z_step(self, observations, steps, muscle_states=None):
        raise NotImplementedError(
            "The learned z step has not been implemented yet."
        )
