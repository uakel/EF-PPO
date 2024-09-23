import numpy as np
import torch
from deprl.vendor.tonic.utils import logger
from deprl.vendor.tonic.torch.agents import Agent
from ef_ppo.models import HLActorCritic 
from deprl.vendor.tonic.torch import models
from deprl.vendor.tonic.torch import normalizers
from ef_ppo import critics as critic_updaters
from deprl.vendor.tonic.torch import updaters
from ef_ppo.hl_segment import HLSegment
from ef_ppo.utils import n_sect
from deprl.dep_controller import DEP

# Defaults
def default_model():
    """
    Returns a default constraint loss actor-critic model.
    """
    return HLActorCritic(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((64, 64), torch.nn.Tanh),
            head=models.DetachedScaleGaussianPolicyHead(),
        ),
        h_critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((64, 64), torch.nn.Tanh),
            head=models.ValueHead(),
        ),
        l_critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP((64, 64), torch.nn.Tanh),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),# 
    )


# Main class    
class EF_PPO(Agent):
    """
    Epigraph Form PPO (EF-PPO).

    EF-PPO: https://arxiv.org/pdf/2305.14154.pdf
    """
    def __init__(self, model=None, replay=None, actor_updater=None,
                 h_critic_updater=None, l_critic_updater=None, log=True,
                 budget_normalizer=1.0):
        """
        Instantiate the agent.
        """
        self.model = model or default_model()
        self.replay = replay or HLSegment()
        self.actor_updater = actor_updater or\
                             updaters.ClippedRatio()
        
        self.h_critic_updater = h_critic_updater or\
                                critic_updaters.VRegression()
        self.l_critic_updater = l_critic_updater or\
                                critic_updaters.VRegression()
        self.log = log
        self.budget_normalizer = budget_normalizer


    def initialize(self, observation_space, action_space, seed=None):
        """
        Initialize the agent.
        """

        # Append the budget dimension to the observation space
        low = np.append(observation_space.low, 0)
        high = np.append(observation_space.high, np.inf)
        observation_space = type(observation_space)(low=low, high=high) 

        # Initialize the model, replay, and updaters.
        super().initialize(seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.actor_updater.initialize(self.model)
        self.h_critic_updater.initialize(self.model.h_critic)
        self.l_critic_updater.initialize(self.model.l_critic)


    def step(self, observations, steps, budgets, muscle_states=None):
        """
        Sample actions from the agent's policy and update 
        the agent's internal state.
        """
        # Sample actions and get their log-probabilities for training.
        actions, log_probs = self._step(observations, budgets)
        actions = actions.numpy(force=True)
        log_probs = log_probs.numpy(force=True)

        # Keep some values for the next update.
        self.last_observations = observations.copy()
        self.last_budgets = budgets.copy()
        self.last_actions = actions.copy()
        self.last_log_probs = log_probs.copy()

        return actions


    def _step(self, observations, budgets):
        """
        Step method that returns the actions and log-probs
        """
        # Cast to tensor
        observations = torch.as_tensor(observations, dtype=torch.float32)
        budget = torch.as_tensor(budgets, dtype=torch.float32)

        # Augment observation
        obs_and_budget = self.budget_augmented_obs(observations, budget)

        # Evaluate actor and sample action
        with torch.no_grad():
            distributions = self.model.actor(obs_and_budget)
            if hasattr(distributions, "sample_with_log_prob"):
                actions, log_probs = distributions.sample_with_log_prob()
            else:
                actions = distributions.sample()
                log_probs = distributions.log_prob(actions)
            log_probs = log_probs.sum(dim=-1)

        return actions, log_probs


    def deterministic_step(self,
                           observations,
                           budgets):
        """
        Step method that returns the mode of the action distribution
        """
        # Cast to tensor
        observations = torch.as_tensor(observations, dtype=torch.float32)
        budget = torch.as_tensor(budgets, dtype=torch.float32)

        # Augment observation
        obs_and_budget = self.budget_augmented_obs(observations, budget)

        # Evaluate actor and sample action
        with torch.no_grad():
            distributions = self.model.actor(obs_and_budget)
            actions = distributions.mode

        return actions

    def deterministic_opt_step(self, observations, steps, muscle_states=None):
        """
        Optimal step using the mode
        """
        # Cast to 2d
        observations = np.atleast_2d(observations)

        # Check if max budget atribute has been set
        if not hasattr(self, 'max_budget'):
            logger.log("WARNING! the attribute 'max_budget' has not been set. "
                       "Please set it before uning the determinisitic_opt_step"
                       " method. Using zero as max_budget..")
            self.max_budget = 0
        
        # Get the maximum z by performing n-section of the value function
        fixed_obs_value = lambda budget: self._compute_v_total(
            np.repeat(observations, len(budget), axis=0),
            budget
        )
        budget_star = n_sect(fixed_obs_value, 
                             -self.max_budget,
                             self.max_budget)
        budget_star = np.array([budget_star])

        # Return the action
        return self.deterministic_step(observations, budget_star)


    def test_step(self, observations, steps, muscle_states=None):
        """
        Wrapper for _test_step
        """
        return self._test_step(observations).numpy(force=True)


    def _test_step(self, observations):
        """
        Return the action from the policy that solves the 
        EF-COCP outer problem
        """
        # Make observations 2d
        observations = np.atleast_2d(observations)

        # Check if max budget atribute has been set
        if not hasattr(self, 'max_budget'):
            logger.log("WARNING! the attribute 'max_budget' has not been set. "
                       "Please set it before uning the determinisitic_opt_step"
                       " method. Using zero as max_budget..")
            self.max_budget = 0

        # Get maximum z by performing n-section of the value function
        fixed_obs_value = lambda budget: self._compute_v_total(
            np.repeat(observations, len(budget), axis=0),
            budget
        )
        budget_star = n_sect(fixed_obs_value, 
                             0,
                             self.max_budget)
        budget_star = np.array([budget_star]) 

        # Return optimal action
        return self._step(observations, budget_star)[0]
    

    def update(self, 
               observations, 
               losses, 
               resets, 
               terminations, 
               const_fn_eval, 
               budgets, 
               steps, 
               fine_tune=False):
        """
        Update the agent's internal state after a transition.
        """

        # Store the last transitions in the replay.
        self.replay.store(
            observations=self.last_observations,
            actions=self.last_actions,
            next_observations=observations,
            losses=losses,
            resets=resets,
            terminations=terminations,
            log_probs=self.last_log_probs,
            const_fn_eval=const_fn_eval,
            budgets=self.last_budgets,
            next_budgets=budgets
        )

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.record(
                np.concatenate(
                    (self.last_observations, 
                     self.last_budgets[:,None] 
                     / self.budget_normalizer), 
                    axis=1
                )
            )
        if self.model.return_normalizer:
            self.model.return_normalizer.record(losses)

        # Update the model if the replay is ready.
        if self.replay.ready():
            self._update(fine_tune=fine_tune)


    def _update(self, fine_tune=False):
        """
        Update the models with the data in the replay buffer.
        """
        # Get a copy of the relevant variables in the buffer
        batch = self.replay.get_full("observations", 
                                     "budgets",
                                     "next_observations",
                                     "next_budgets")

        # Evaluate critics on buffer for GAE bootstraps
        critic_evals = self._evaluate(**batch)
        
        # Cast to numpy array
        to_numpy = lambda x: x.numpy(force=True)
        critic_evals = map(to_numpy, critic_evals)

        # Compute GAE's
        self.replay.compute_GAEs(*critic_evals)

        actor_critic_iterations = 0
        keys = ("observations", "budgets", "actions", "Q_h", 
                "Q_l", "EF_COCP_advantages", "log_probs")

        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys):
            # make batch
            batch = {k: torch.as_tensor(v) for k, v in batch.items()}

            # run updates
            infos = self._update_actor_critic(**batch,
                                              fine_tune=fine_tune)
            actor_critic_iterations += 1

            # log
            if self.log:
                for key in infos:
                    for k, v in infos[key].items():
                        logger.store(key + "/" + k, v.numpy(force=True))
        
        if self.log:
            logger.store("actor_critic/iterations", actor_critic_iterations)

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()


    def budget_augmented_obs(self, observations, budgets):
        """
        Method to create a vector that contains the observation and the budget
        """
        return torch.cat((observations, 
                          budgets[:, None] / 
                          self.budget_normalizer), dim=1)


    def _evaluate(self, observations, budgets, next_observations, next_budgets):
        """
        Evaluate the critics on the buffer
        """
        # Cast to tensor
        observations = torch.as_tensor(observations, dtype=torch.float32)
        budgets = torch.as_tensor(budgets, dtype=torch.float32)

        next_observations = torch.as_tensor(
            next_observations, dtype=torch.float32)
        next_budgets = torch.as_tensor(
            next_budgets, dtype=torch.float32)

        # Augment observations with budget
        obs_and_budget = self.budget_augmented_obs(observations,
                                                   budgets)
        next_obs_and_budget = self.budget_augmented_obs(next_observations, 
                                                        next_budgets)
        # Do a forward pass through l and h critics     
        with torch.no_grad():
            l_values = self.model.l_critic(obs_and_budget)
            next_l_values = self.model.l_critic(next_obs_and_budget)
            h_values = self.model.h_critic(obs_and_budget)
            next_h_values = self.model.h_critic(next_obs_and_budget)

        return l_values, next_l_values, h_values, next_h_values


    def _compute_v_total(self, observations, budgets):
        """
        Compute the value function of the EF-COCP
        """
        observations = torch.as_tensor(observations, dtype=torch.float32) 
        t_budgets = torch.as_tensor(budgets, dtype=torch.float32) 

        obs_and_budget = self.budget_augmented_obs(observations, t_budgets)
        with torch.no_grad():
            h_values = self.model.h_critic(
                obs_and_budget).numpy(force=True).flatten()
            l_values = self.model.l_critic(
                obs_and_budget).numpy(force=True).flatten()
        return np.maximum(h_values, l_values - budgets)


    def _update_actor_critic(
        self, 
        observations,
        budgets,
        actions,
        Q_h,
        Q_l,
        EF_COCP_advantages,
        log_probs,
        fine_tune=False
    ):
        """
        Update the actor and the critics
        """
        obs_and_budget = self.budget_augmented_obs(observations, budgets)
        h_critic_infos = self.h_critic_updater(obs_and_budget, Q_h)
        l_critic_infos = self.l_critic_updater(obs_and_budget, Q_l)
        if not fine_tune:
            actor_infos = self.actor_updater(
                obs_and_budget, 
                actions, 
                -EF_COCP_advantages,
                log_probs
            )
        return dict(actor=None if fine_tune else actor_infos, 
                    h_critic=h_critic_infos,
                    l_critic=l_critic_infos)

    def _load_weights(self, path): 
        """
        Load weights 
        """
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)

# DEP-RL EF-PPO
class DEP_EF_PPO(EF_PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expl = self.dep = DEP()
        self.since_switch = 500

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(observation_space, action_space, seed)
        self.dep.initialize(observation_space, action_space, seed)

    def step(
        self, observations, steps, budgets, muscle_states=None, greedy_episode=False
    ):
        if greedy_episode:
            return super().step(
                observations, steps, budgets, muscle_states
            )
        self.since_switch += 1
        dep_actions = self.dep.step(muscle_states, steps)
        if self.since_switch > self.dep.intervention_length:
            if np.random.uniform() < self.dep.intervention_proba:
                self.since_switch = 0
            return super().step(
                observations, steps, budgets, muscle_states
            )
        self.last_observations = observations.copy()
        self.last_actions = dep_actions.copy()
        return dep_actions