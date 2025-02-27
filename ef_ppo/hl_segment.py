from deprl.vendor.tonic.replays.segments import Segment
import numpy as np
from deprl.vendor.tonic.replays.utils import flatten_batch

class HLSegment(Segment):
    """
    Segment replay buffer for EF-PPO
    """
    def __init__(
        self,
        size=128,
        batch_iterations=5,
        batch_size=None,
        discount_factor=0.97,
        trace_decay=0.95,
        h_term_penalty=0,
        l_term_penalty=0,
    ):
        self.steps_before_batches = -1
        self.trace_decay_sum_weights = np.array(
            [trace_decay**i for i in range(size)]
        )[::-1]
        self.h_term_penalty = h_term_penalty
        self.l_term_penalty = l_term_penalty

        super().__init__(
            size=size,
            batch_iterations=batch_iterations,
            batch_size=batch_size,
            discount_factor=discount_factor,
            trace_decay=trace_decay
        )

    def get_keys(self, *keys):
        """
        Get data without resetting the index
        """
        return {k: flatten_batch(self.buffers[k]) for k in keys} # type: ignore


    def compute_GAEs(
        self,
        l_bootstrap,
        next_r_bootstrap,
        h_bootstrap,
        next_h_bootstrap
    ):
        """
        Compute the generalized advantage estimates for the modified value fkt
        """
        # Get buffer characteristics 
        shape = self.buffers["rewards"].shape
        num_workers = shape[1] 

        # Reshape and save bootstraps in buffer
        self.buffers["h_bootstrap"] = h_bootstrap \
                                    = h_bootstrap.reshape(shape)
        self.buffers["next_h_bootstrap"] = next_h_bootstrap \
                                         = next_h_bootstrap.reshape(shape)
        self.buffers["l_bootstrap"] = l_bootstrap \
                                    = l_bootstrap.reshape(shape)
        self.buffers["next_l_bootstrap"] = next_r_bootstrap \
                                         = next_r_bootstrap.reshape(shape)

        # Define array holding the lambda-return style
        # estimates of action-value functions
        Q_h = np.zeros(shape, dtype=np.float32) 
        Q_r = np.zeros(shape, dtype=np.float32) 
        Q_tot = np.zeros(shape, dtype=np.float32)

        # Get the constraint function evaluations and rewards
        const_fn_evals = self.buffers["const_fn_eval"]
        rewards = self.buffers["rewards"]

        # Get budgets
        budgets = self.buffers["budgets"]

        # Get resets and terminations
        resets = self.buffers["resets"].astype(bool)
        terminations = self.buffers["terminations"]

        # Initialize n-step estimates
        n_step_Q_h_estimates = np.zeros(shape, dtype=np.float32)
        n_step_Q_r_estimates = np.zeros(shape, dtype=np.float32)
        
        # Initialize sum coefficients
        sum_coefficients = np.zeros(shape, dtype=np.float32)

        # Computation of the n-step action-value estimates
        it = 0
        it_since_reset = np.zeros(shape[1], dtype=int)
        for t in range(len(const_fn_evals) - 1, -1, -1): # iterate backwards
            # Reset iteration counter if reset occurred
            it_since_reset[resets[t]] = 0

            # Place bootstrap values at iteration index in n-step estimate
            # arrays and apply termination penalty if termination occurred
            n_step_Q_h_estimates[it] = next_h_bootstrap[t]
            n_step_Q_h_estimates[it] *= (1 - terminations[t].astype(int))
            n_step_Q_h_estimates[it] += terminations[t].astype(int) * \
                self.h_term_penalty
            n_step_Q_r_estimates[it] = next_r_bootstrap[t]
            n_step_Q_r_estimates[it] *= (1 - terminations[t].astype(int))
            n_step_Q_r_estimates[it] += terminations[t].astype(int) * \
                self.l_term_penalty

            # Use recursive rule to calculate the n-step estimates from
            # last iterations n-step estimates
            n_step_Q_h_estimates = self._h_reduce(const_fn_evals[t],
                                                  n_step_Q_h_estimates)
            n_step_Q_r_estimates = self._r_reduce(rewards[t],
                                                  n_step_Q_r_estimates)
            n_step_Q_tot_estimates = self._q_tot_map(n_step_Q_h_estimates,
                                                     n_step_Q_r_estimates,
                                                     budgets[t])

            # Generate sum coefficients
            sum_coefficients[:, :] = 0.0
            for w in range(num_workers):
                sum_coefficients[it - it_since_reset[w]:
                                 it + 1, w] \
                    = self.trace_decay_sum_weights[-it_since_reset[w] - 1:]

            # Calculate convex combination of n-step estimates for
            # lambda-return style action value function estimates
            normalization = np.sum(sum_coefficients, axis=0)
            Q_h[t] = np.sum(
                sum_coefficients * n_step_Q_h_estimates, 
                axis=0
            ) / normalization
            Q_r[t] = np.sum(
                sum_coefficients * n_step_Q_r_estimates,
                axis=0
            ) / normalization 
            Q_tot[t] = np.sum(
                sum_coefficients * n_step_Q_tot_estimates,
                axis=0
            ) / normalization

            # Increase index variables
            it += 1
            it_since_reset += 1


        self.buffers["Q_h"] = Q_h
        self.buffers["Q_r"] = Q_r
        self.buffers["Q_tot"] = Q_tot
        self.buffers["EF_COCP_advantages"] = \
            Q_tot - self._base_line(h_bootstrap, l_bootstrap, budgets)
        # self.buffers["EF_COCP_advantages"] *= -1

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

class SumHLSegment(HLSegment):
    def __init__(
        self,
        max_violation=0,
        size=128,
        batch_iterations=5,
        batch_size=None,
        discount_factor=0.97,
        trace_decay=0.95,
        h_term_penalty=0,
        l_term_penalty=0
    ):
        self.max_violation = max_violation
        super().__init__(size, batch_iterations, batch_size, discount_factor, trace_decay, h_term_penalty, l_term_penalty)

    def _h_reduce(
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
        return np.minimum(self.max_violation - q_h_estimates, q_l_estimates + budgets)

    def _base_line_map(
        self,
        v_h_estimates,
        v_l_estimates,
        budgets,
    ):
        return np.minimum(self.max_violation - v_h_estimates, v_l_estimates + budgets)

