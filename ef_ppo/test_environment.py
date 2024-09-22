import numpy as np

from deprl.vendor.tonic import logger


def test_mujoco(env, agent, steps, constraint_function, params=None, test_episodes=10):
    """
    Tests the agent on the test environment.
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
    for _ in range(test_episodes):
        metrics = {
            "test/episode_score": 0,
            "test/episode_length": 0,
            "test/effort": 0,
            "test/terminated": 0,
            "test/constraint_score": 0,
        }
        if eval_rwd_metrics:
            rwd_metrics = {k: [] for k in env.environments[0].rwd_dict.keys()}

        while True:
            # Select an action.
            actions = agent.test_step(env.test_observations, steps)
            assert not np.isnan(actions.sum())
            logger.store("test/action", actions, stats=True)

            # Take a step in the environment.
            env.test_observations, _, info = env.step(actions)
            const_eval = constraint_function(env.test_observations, None)

            # Update metrics
            metrics["test/episode_score"] += info["rewards"][0]
            metrics["test/constraint_score"] += const_eval[0]
            metrics["test/episode_length"] += 1

            if env.environments[0].sim.model.na > 0:
                metrics["test/effort"] += np.mean(
                    np.square(env.environments[0].unwrapped.sim.data.act)
                )
            metrics["test/terminated"] += int(info["terminations"])
            if eval_rwd_metrics:
                for k, v in env.environments[0].rwd_keys_wt.items():
                    rwd_metrics[k].append(v * env.environments[0].rwd_dict[k])

            if info["resets"][0]:
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
    return metrics
