import random
from typing import BinaryIO, Dict, List, Optional, Union

import numpy as np
import torch

from  .camera import extract_camera_params
import open3d as o3d


def get_point_clouds(cameras, depths, alphas, rgbs=None):
    """
    Generates a 3D point cloud from camera parameters, depth maps, and optional RGB colors.

    Args:
        cameras: Camera intrinsics and extrinsics.
        depths: Depth maps of shape (N, H, W), where N is the number of images.
        alphas: Binary mask indicating valid depth points.
        rgbs: Optional RGB color values corresponding to depth points.

    Returns:
        PointCloud: A structured point cloud representation with 3D coordinates and color information.
    """
    Hs, Ws, intrinsics, c2ws = extract_camera_params(cameras)
    W, H = int(Ws[0].item()), int(Hs[0].item())
    assert (depths.shape == alphas.shape)
    coords = []
    rgbas = []

    N = intrinsics.shape[0]
    
    # TODO: Compute ray origins and directions for each pixel
    # Hint: You need to use the camera intrinsics (intrinsics) and extrinsics (c2ws)
    # to convert pixel coordinates into world-space rays.
    # rays_o, rays_d = ......
    

    device = torch.device('cuda')

    rays_o_c = torch.tensor([0, 0, 0, 1], dtype=torch.double)         
    rays_o_c = rays_o_c.repeat(N, 1)
    rays_o_c = rays_o_c.unsqueeze(-1).to(device)   # Shape: (200, 4, 1)
    
    # c2ws_inv = torch.linalg.inv(c2ws).to(dtype=torch.double)
    # c2ws_inv = torch.linalg.inv(c2ws)
    
    # rays_o = torch.bmm(c2ws_inv, rays_o_c)  # [200, 4, 1], one for each image

    
    
    w_coords = torch.arange(W).float()  # [0, 1, 2, ..., W-1]
    h_coords = torch.arange(H).float()  # [0, 1, 2, ..., H-1]

    # Create 2D coordinate grids
    grid_w, grid_h = torch.meshgrid(w_coords, h_coords, indexing='xy')  # H, W
    grid_one = torch.ones(H, W)
    pix_coords = torch.stack([grid_w, grid_h, grid_one], dim=-1)
    pix_coords = pix_coords.unsqueeze(0).unsqueeze(-1).expand(N, -1, -1, -1, 1).to(device)

    intrinsics = intrinsics[:, :3, :3]
    
    intrinsics_inv = torch.linalg.inv(intrinsics)
    intrinsics_inv = intrinsics_inv.unsqueeze(1).unsqueeze(2)
    intrinsics_inv = intrinsics_inv.expand(-1, H, W, -1, -1)
    
    rays_d_c = torch.matmul(intrinsics_inv, pix_coords)   # [200, 6, 10, 3, 1] # [X/Z, Y/Z, 1] in cam frame
    rays_d_c = rays_d_c.squeeze(-1)
    # print('rays_d_c', rays_d_c.shape)
    

    # TODO: Compute 3D world coordinates using depth values
    # Hint: Use the ray equation: P = O + D * depth
    # P: 3D point, O: ray origin, D: ray direction, depth: depth value
    
    depths = depths.unsqueeze(-1)
    
    points_c = rays_d_c * depths
    
    points_c = torch.cat((points_c, torch.ones(N, H, W, 1).to(device)), dim=-1)
    points_c = points_c.unsqueeze(-1)

    
    c2ws = c2ws.unsqueeze(1).unsqueeze(2)
    c2ws = c2ws.expand(-1, H, W, -1, -1)

    points = torch.matmul(c2ws, points_c)   # (Xw, Yw, Zw, 1)
    points = points.squeeze(-1)
    points = points[:, :, :, :3]
    

    # TODO: Apply the alpha mask to filter valid points
    # Hint: Mask should be applied to both coordinates and RGB values (if provided)
    # mask = ......
    # coords = pts[mask].cpu().numpy()

    coords = points[alphas.to(dtype=bool)].cpu().numpy()
    print('filtered coords', coords.shape)  # (9815241, 3)

    
    rgbas = torch.cat((rgbs, alphas.unsqueeze(-1).to(device)), dim=-1)
    rgbas = rgbas[alphas.to(dtype=bool)].cpu().numpy()
    print('filtered rgbas', rgbas.shape)

    visualize(coords, rgbas)
    
    if rgbs is not None:
        channels = dict(
            R=rgbas[..., 0],
            G=rgbas[..., 1],
            B=rgbas[..., 2],
            A=rgbas[..., 3],
        )
    else:
        channels = {}

    point_cloud = PointCloud(coords, channels)
    return point_cloud

def visualize(coords, rgbas):


    import numpy as np
    import plotly.graph_objects as go

    # Normalize RGB
    if rgbas[:, :3].max() > 1.0:
        rgb = rgbas[:, :3] / 255.0
    else:
        rgb = rgbas[:, :3]

    # Convert to Plotly-compatible color strings
    rgb_str = ['rgba({},{},{},{})'.format(int(r*255), int(g*255), int(b*255), a)
            for (r, g, b), a in zip(rgb, rgbas[:, 3])]

    # Sample if too many points
    max_points = 100000
    if len(coords) > max_points:
        idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[idx]
        rgb_str = [rgb_str[i] for i in idx]

    # Create scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=rgb_str,
        )
    ))
    fig.update_layout(scene=dict(aspectmode='data'))

    x_max = 0.8
    x_min = -0.8
    y_max = 1.5
    y_min = -1.5
    z_max = 1
    z_min = -0.4
    # Control the visible range of each axis
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            zaxis=dict(range=[z_min, z_max]),
            # aspectmode='data'
            aspectmode='manual',
            aspectratio=dict(
            x=(x_max - x_min),
            y=(y_max - y_min),
            z=(z_max - z_min),
        )
        )
    )
    print('showing figure')
    fig.show()




def preprocess(data, channel):
    if channel in ["R", "G", "B", "A"]:
        return np.round(data * 255.0)
    return data


class PointCloud:
    def __init__(self, coords: np.ndarray, channels: Dict[str, np.ndarray]) -> None:
        self.coords = coords
        self.channels = channels

    def __repr__(self) -> str:
        str = f"coords:{len(self.coords)} \t channels:{list(self.channels.keys())}"
        return str

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return PointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"},
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with open(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx
            cur_dists = np.minimum(cur_dists, compute_dists(idx))
        return self.subsample(indices, **subsample_kwargs)

    def subsample(self, indices: np.ndarray, average_neighbors: bool = False) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
            )

        new_coords = self.coords[indices]
        neighbor_indices = PointCloud(coords=new_coords, channels={}).nearest_points(self.coords)

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(coords=new_coords, channels=new_channels)

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack([preprocess(self.channels[name], name) for name in channel_names], axis=-1)
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            dists = norms + np.sum(batch**2, axis=-1)[:, None] - 2 * (batch @ self.coords.T)
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0) for k, v in self.channels.items()
            },
        )
