import numpy as np

def n_sect(function, x_min, x_max, n_iter=5, n=20):
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
    flips = np.logical_and(evaluations[:-1] > 0, evaluations[1:] <= 0)
    if np.sum(flips) == 0:
        return points[np.argsort(evaluations)[0]]
        logger.store("test/no_sign_flip_outer_problem", 1.0)
    if n_iter == 0:
        return min(points[:-1][flips][0], points[1:][flips][0])
    return n_sect(function, 
                  points[:-1][flips][0],
                  points[1:][flips][0], 
                  n_iter=n_iter-1, 
                  n=n) 
