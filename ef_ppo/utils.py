import numpy as np

def discounted_cost_score(costs, discount):
    """
    Calculates the discounted score.
    """
    scores = np.zeros_like(costs)
    carry = 0
    for i in range(len(costs) - 1, -1, -1):
        carry = carry * discount + costs[i]
        scores[i] = carry
    return scores

def discounted_constraint_score(constraint_fn_evals, discount):
    """
    Calculates the discounted constraint score.
    """
    scores = np.zeros_like(constraint_fn_evals)
    carry = 0
    for i in range(len(constraint_fn_evals) - 1, -1, -1):
        carry = max(constraint_fn_evals[i], 
                    (1 - discount) * constraint_fn_evals[i] 
                    + discount * carry)
        scores[i] = carry
    return scores

def n_sect(function, x_min, x_max, n_iter=2, n=20):
    """
    parallel scalar root finding

    Args:
        function: A function that computes 0th dimension in parallel
        x_min: point to start n-setion from
        x_max: point where n-section ends
        n-iter: iteration number
        n: number of parallel evaluations
    """
    points = np.linspace(x_min, x_max, n)
    evaluations = function(points).flatten()
    if all(evaluations >= 0):
        return points[np.argmin(evaluations)]
    if all(evaluations <= 0):
        return x_min
    down_flips = (evaluations[:-1] >= 0) & (evaluations[1:] < 0)
    if sum(down_flips) == 0:
        return x_max
    if n_iter == 0:
        return points[1:][down_flips][0]
    return n_sect(function, 
                  points[:-1][down_flips][0],
                  points[1:][down_flips][0], 
                  n_iter=n_iter-1, 
                  n=n) 
