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

def find_root(function, x_min, x_max, n=64):
    """
    parallel scalar root finding

    Args:
        function: function
        x_min: np.array of dim (dim,)
        x_max: np.array of dim (dim,)
        n: number of samples

    Returns:
        np.array of dim (dim,)
    """
    search_space = np.linspace(x_min, x_max, n, axis=-1)
    evals = function(search_space)
    min_distance_indices = np.argmin(np.abs(evals), axis=-1, keepdims=True)
    greater_than_zero = evals > -0.05
    np.put_along_axis(greater_than_zero, min_distance_indices, True, axis=-1)
    best_indices = np.argmax(greater_than_zero, axis=-1, keepdims=True)
    z_star = np.take_along_axis(search_space, best_indices, axis=-1)[..., 0]

    # Old..
    # search_space = np.linspace(x_min, x_max, n, axis=-1)
    # evals = function(search_space)

    # indices = np.argmin(np.abs(evals), axis=-1, keepdims=True)
    # z_star = np.take_along_axis(search_space, indices, axis=-1)[..., 0]
    
    # Debug
    # import matplotlib.pyplot as plt
    # plt.plot(search_space, evals)
    # plt.plot([z_star, z_star], [np.min(evals), np.max(evals)], 'k--', label='z_star')
    # plt.plot([x_min, x_max], [0, 0], 'k--', label='y=0')
    # plt.legend()
    # plt.show()

    return z_star



