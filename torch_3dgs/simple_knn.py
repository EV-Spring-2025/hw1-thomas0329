import torch


def compute_mean_knn_dist(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Computes the mean k-nearest neighbor (k-NN) distance for each point in a point cloud.

    Args:
        points (torch.Tensor): A tensor of shape (N, D), where N is the number of points 
                               and D is the dimensionality (e.g., 3 for 3D points).
        k (int): The number of nearest neighbors to consider.

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the mean distance to the k nearest neighbors for each point.
    """

    diffs = points[:, None, :] - points[None, :, :]
    dists_sq = torch.sum(diffs ** 2, dim=-1)

    print('points', points.shape)
    print('diffs', diffs.shape)
    # TODO: Ignore self-distances (distance of each point to itself)
    # Hint: Use fill_diagonal_ to set diagonal values to infinity
    # dists_sq.fill_diagonal_(......)


    # TODO: Find k-nearest neighbors
    # Hint: Use torch.topk to find the smallest k distances for each point
    # knn_vals, _ = ......

    # TODO: Compute the mean k-NN distance
    # Hint: Average the distances over the k neighbors
    # mean_knn_dist = ......
    
    return mean_knn_dist
