import torch




def compute_variance_waypoints(trajs_pos):
    assert trajs_pos.ndim == 3  # batch, horizon, state_dim

    sum_var_waypoints = 0.
    for via_points in trajs_pos.permute(1, 0, 2):  # horizon, batch, position
        parwise_distance_between_points_via_point = torch.cdist(via_points, via_points, p=2)
        distances = torch.triu(parwise_distance_between_points_via_point, diagonal=1).view(-1)
        sum_var_waypoints += torch.var(distances)
    return sum_var_waypoints


def compute_smoothness_from_vel(trajs_vel):
    assert trajs_vel.ndim == 3

    smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1)
    smoothness = smoothness.sum(-1)  # sum over trajectory horizon
    return smoothness

def compute_smoothness_from_pos(trajs_pos, trajs_vel=None):
    """
    traj_pos: batch, horizon, state_dim
    """
    assert trajs_pos.ndim == 3

    trajs_vel = torch.diff(trajs_pos, dim=-2)
    smoothness = torch.linalg.norm(torch.diff(trajs_vel, dim=-2), dim=-1)
    smoothness = smoothness.sum(-1)  # sum over trajectory horizon
    return smoothness