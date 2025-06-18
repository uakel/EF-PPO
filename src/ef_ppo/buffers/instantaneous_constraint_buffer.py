import numpy as np
from .base import AbstractHLSegmentBuffer

class InstantaneousConstraintSegmentBuffer(AbstractHLSegmentBuffer):
    """
    Segment replay buffer for EF-PPO
    """
    def _transform_budgets(
        self,
        budgets,
        resets,
        rewards,
        const_fn_evals,
    ):
        return budgets

    def _sum_reduce(
        self,
        evals,
        estimates,
    ):
        return evals + self.discount_factor * estimates

    def _min_reduce(
        self,
        evals,
        estimates,
    ):
        g = self.discount_factor
        return np.minimum(-evals, -(1 - g) * evals + g * estimates)

    def _h_reduce(
        self,
        evals,
        estimates,
    ):
        return self._min_reduce(evals, estimates)

    def _r_reduce(
        self,
        evals,
        estimates,
    ):
        return self._sum_reduce(evals, estimates)

    def _q_tot_map(
        self,
        q_h_estimates,
        q_l_estimates,
        budgets,
    ):
        return np.minimum(q_h_estimates, q_l_estimates + budgets)
    
    def _base_line(
        self,
        v_h_estimates,
        v_l_estimates,
        budgets,
    ):
        return np.minimum(v_h_estimates, v_l_estimates + budgets)

