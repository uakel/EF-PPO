import numpy as np

from ef_ppo import logger
from ef_ppo.utils import discounted_cost_score, discounted_constraint_score

def test(env, 
         agent, 
         steps, 
         constraint_function, 
         params=None, 
         test_episodes=10, 
         data_path=lambda env: env.environments[0].unwrapped.sim.data):
    """
    Tests the EF-PPO agent on the test environment.
    """
    # Start the environment.
    if not hasattr(env, "test_observations"):
        # Dont use dep in evaluation
        env.test_observations, _ = env.start()
        assert len(env.test_observations) == 1

    eval_rwd_metrics = (
        True if hasattr(env.environments[0], "rwd_dict") else False
    )

    # Test loop.
    for ep_index in range(test_episodes):
        metrics = {
            "test/cost/undiscounted_cost_score": 0,
            "test/constraint/undiscounted_constraint_score": 0,
            "test/episode_length": 0,
            "test/effort": 0,
            "test/terminated": 0,
        }
        if eval_rwd_metrics:
            rwd_metrics = {k: [] for k in env.environments[0].rwd_dict.keys()}

        costs_since_reset = []
        constraint_function_evaluations_since_reset = []
        while True:
            # Select an action.
            actions, budget_star = agent.test_step(env.test_observations, steps)
            assert not np.isnan(actions.sum())
            logger.store("test/action", actions, stats=True)
            logger.store("test/budget_star", budget_star, stats=True)

            # Take a step in the environment.
            env.test_observations, _, info = env.step(actions)
            info["costs"] = info.pop("rewards")

            # Get and log constraint function evaluations
            const_fn_eval = constraint_function(env.test_observations, None)
            logger.store("test/constraint_function_evaluations", 
                         const_fn_eval, stats=True)

            # Update metrics
            metrics["test/cost/undiscounted_cost_score"] += info["costs"][0]
            metrics["test/constraint/undiscounted_constraint_score"]\
                = np.maximum(const_fn_eval[0], 
                             metrics["test/constraint/undiscounted_constraint_score"])
            metrics["test/episode_length"] += 1

            # Save the cost and constraint function evaluations in temporary lists
            constraint_function_evaluations_since_reset.append(const_fn_eval[0])
            costs_since_reset.append(info["costs"][0])

            # log qpos, muscle_activity, lengths forces and constraint function evaluations
            # for the first 5 episodes
            measurements = dict(
                qpos = env.environments[0].qpos(),
                act = env.environments[0].muscle_activity(),
                lengths = env.environments[0].muscle_lengths(),
                forces = env.environments[0].muscle_forces(),
            )
            if ep_index < 5:
                logger.store(f"test/rollout_litterals/constraint_function_evaluations/ep_{ep_index}",
                             list(const_fn_eval), raw=True, print=False)
                logger.store(f"test/rollout_litterals/costs/ep_{ep_index}",
                             list(info["costs"]), raw=True, print=False)
                logger.store(f"test/rollout_litterals/budget_star_raw/ep_{ep_index}",
                             list(budget_star), raw=True, print=False)
                for quant, values in measurements.items():
                    for i, value in enumerate(values):
                        logger.store(f"test/rollout_litterals/{quant}/{str(i)}/ep_{ep_index}", 
                                     value, raw=True, print=False)

            # Get and log cost
            cost = info["costs"]
            logger.store("test/cost/environment_costs", cost, stats=True)


            # Save effort
            metrics["test/effort"] += np.mean(
                np.square(data_path(env).act)
            )
            metrics["test/terminated"] += int(info["terminations"])
            if eval_rwd_metrics:
                for k, v in env.environments[0].rwd_keys_wt.items():
                    rwd_metrics[k].append(v * env.environments[0].rwd_dict[k])

            if info["resets"][0]:
                # Log discounted cost and constraint function scores
                # costs
                discounted_cost_scores = discounted_cost_score(
                    costs_since_reset,
                    agent.replay.discount_factor,
                )
                for score in discounted_cost_scores:
                    logger.store("test/cost/discounted_cost_score", score, stats=True)
                # constraints
                discounted_constraint_scores = discounted_constraint_score(
                    constraint_function_evaluations_since_reset,
                    agent.replay.discount_factor,
                )
                for score in discounted_constraint_scores:
                    logger.store("test/constraint/discounted_constraint_score", score, stats=True)
                break

        # Log the data.Average over episode length here
        metrics["test/terminated"] /= metrics["test/episode_length"]
        metrics["test/effort"] /= metrics["test/episode_length"]
        if eval_rwd_metrics:
            for k, v in rwd_metrics.items():
                metrics["test/rwd_metrics/" + k] = np.sum(v)
        # average over episodes in logger
        for k, v in metrics.items():
            logger.store(k, v, stats=True)
