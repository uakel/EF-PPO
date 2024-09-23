from deprl.vendor.tonic.replays.segments import Segment
import numpy as np

class HLSegment(Segment):
    """
    Segment replay buffer for EF-PPO
    """
    def __init__(
        self,
        size=1028,
        batch_iterations=5,
        batch_size=None,
        discount_factor=0.97,
        trace_decay=0.95,
        h_term_penalty=None,
        l_term_penalty=None
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


    def compute_GAEs(
        self,
        l_bootstrap,
        next_l_bootstrap,
        h_bootstrap,
        next_h_bootstrap
    ):
        # Get buffer characteristics 
        shape = self.buffers["losses"].shape
        num_workers = shape[1] 

        # Reshape and save bootstraps in buffer
        self.buffers["h_bootstrap"] = h_bootstrap \
                                    = h_bootstrap.reshape(shape)
        self.buffers["next_h_bootstrap"] = next_h_bootstrap \
                                         = next_h_bootstrap.reshape(shape)
        self.buffers["l_bootstrap"] = l_bootstrap \
                                    = l_bootstrap.reshape(shape)
        self.buffers["next_l_bootstrap"] = next_l_bootstrap \
                                         = next_l_bootstrap.reshape(shape)

        # Define array holding the lambda-return style
        # estimates of action-value functions
        Q_h = np.zeros(shape, dtype=np.float32) 
        Q_l = np.zeros(shape, dtype=np.float32) 
        Q_tot = np.zeros(shape, dtype=np.float32)

        # Get the constraint function evaluations and losses
        const_fn_evals = self.buffers["const_fn_eval"]
        losses = self.buffers["losses"]

        # Get budgets
        budgets = self.buffers["budgets"]

        # Get resets and terminations
        resets = self.buffers["resets"].astype(bool)
        terminations = self.buffers["terminations"]
        h_term_penalty = self.h_term_penalty if self.h_term_penalty \
                                             is not None \
                                             else np.max(h_bootstrap) * 1.5
        l_term_penalty = self.l_term_penalty if self.l_term_penalty \
                                             is not None \
                                             else np.max(l_bootstrap) * 1.5

        # Initialize n-step estimates
        n_step_Q_h_estimates = np.zeros(shape, dtype=np.float32)
        n_step_Q_l_estimates = np.zeros(shape, dtype=np.float32)
        
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
                h_term_penalty
            n_step_Q_l_estimates[it] = next_l_bootstrap[t]
            n_step_Q_l_estimates[it] *= (1 - terminations[t].astype(int))
            n_step_Q_l_estimates[it] += terminations[t].astype(int) * \
                l_term_penalty

            # Use recursive rule to calculate the n-step estimates from
            # last iterations n-step estimates
            n_step_Q_h_estimates = np.maximum(
                const_fn_evals[t], 
                (1 - self.discount_factor) * const_fn_evals[t] +
                self.discount_factor * n_step_Q_h_estimates 
            )
            n_step_Q_l_estimates = losses[t] + \
                self.discount_factor * n_step_Q_l_estimates 
            n_step_Q_tot_estimates = np.maximum(n_step_Q_h_estimates, 
                n_step_Q_l_estimates - budgets[t])

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
            Q_l[t] = np.sum(
                sum_coefficients * n_step_Q_l_estimates,
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
        self.buffers["Q_l"] = Q_l
        self.buffers["Q_tot"] = Q_tot
        self.buffers["EF_COCP_advantages"] = \
            Q_tot - np.maximum(h_bootstrap, l_bootstrap - budgets)

