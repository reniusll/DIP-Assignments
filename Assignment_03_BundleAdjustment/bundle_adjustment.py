from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BundleAdjustmentData:
    observations: torch.Tensor
    visibility: torch.Tensor
    colors: np.ndarray
    view_names: list[str]
    image_width: int
    image_height: int

    @property
    def cx(self) -> float:
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        return self.image_height / 2.0

    @property
    def num_views(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_points(self) -> int:
        return int(self.observations.shape[1])


def load_bundle_adjustment_data(data_dir: str | Path) -> BundleAdjustmentData:
    data_dir = Path(data_dir)
    points2d = np.load(data_dir / "points2d.npz")
    view_names = sorted(points2d.files)
    stacked = np.stack([points2d[name] for name in view_names], axis=0).astype(np.float32)

    observations = torch.from_numpy(stacked[..., :2])
    visibility = torch.from_numpy(stacked[..., 2] > 0.5)
    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32)

    return BundleAdjustmentData(
        observations=observations,
        visibility=visibility,
        colors=colors,
        view_names=view_names,
        image_width=1024,
        image_height=1024,
    )


def inverse_softplus(x: float) -> float:
    if x > 20.0:
        return float(x)
    return float(np.log(np.expm1(x)))


def euler_xyz_to_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
    x, y, z = euler_angles.unbind(dim=-1)
    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    r00 = cy * cz
    r01 = -cy * sz
    r02 = sy
    r10 = sx * sy * cz + cx * sz
    r11 = -sx * sy * sz + cx * cz
    r12 = -sx * cy
    r20 = -cx * sy * cz + sx * sz
    r21 = cx * sy * sz + sx * cz
    r22 = cx * cy

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(values.dtype)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (values * mask).sum(dim=dim) / denom


def initialize_points_from_observations(
    observations: torch.Tensor,
    visibility: torch.Tensor,
    focal_init: float,
    depth_init: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    u = observations[..., 0]
    v = observations[..., 1]
    vis = visibility

    x_candidates = (u - cx) * depth_init / focal_init
    y_candidates = -(v - cy) * depth_init / focal_init

    x0 = _masked_mean(x_candidates, vis, dim=0)
    y0 = _masked_mean(y_candidates, vis, dim=0)

    visibility_ratio = vis.float().mean(dim=0)
    z0 = (0.25 - visibility_ratio).clamp(min=-0.25, max=0.25)

    points = torch.stack([x0, y0, z0], dim=-1)
    points += 0.01 * torch.randn_like(points)
    return points


def initialize_translations(
    observations: torch.Tensor,
    visibility: torch.Tensor,
    focal_init: float,
    depth_init: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    u = observations[..., 0]
    v = observations[..., 1]
    vis = visibility

    u_center = _masked_mean(u, vis, dim=1)
    v_center = _masked_mean(v, vis, dim=1)

    tx = (u_center - cx) * depth_init / focal_init
    ty = -(v_center - cy) * depth_init / focal_init
    tz = torch.full_like(tx, -depth_init)

    return torch.stack([tx, ty, tz], dim=-1)


class BundleAdjustmentModel(nn.Module):
    def __init__(
        self,
        data: BundleAdjustmentData,
        focal_init: float = 900.0,
        depth_init: float = 2.5,
    ) -> None:
        super().__init__()
        self.image_width = float(data.image_width)
        self.image_height = float(data.image_height)
        self.cx = float(data.cx)
        self.cy = float(data.cy)
        self.depth_init = float(depth_init)
        self.focal_init = float(focal_init)

        points3d_init = initialize_points_from_observations(
            observations=data.observations,
            visibility=data.visibility,
            focal_init=focal_init,
            depth_init=depth_init,
            cx=self.cx,
            cy=self.cy,
        )
        translations_init = initialize_translations(
            observations=data.observations,
            visibility=data.visibility,
            focal_init=focal_init,
            depth_init=depth_init,
            cx=self.cx,
            cy=self.cy,
        )

        initial_yaw = torch.linspace(-0.25, 0.25, data.num_views)
        euler_init = torch.zeros(data.num_views, 3, dtype=torch.float32)
        euler_init[:, 1] = initial_yaw

        self.raw_focal = nn.Parameter(torch.tensor(inverse_softplus(focal_init), dtype=torch.float32))
        self.euler_angles = nn.Parameter(euler_init)
        self.translations = nn.Parameter(translations_init)
        self.points3d = nn.Parameter(points3d_init)

    @property
    def focal(self) -> torch.Tensor:
        return F.softplus(self.raw_focal) + 1.0

    def project(self) -> tuple[torch.Tensor, torch.Tensor]:
        rotation = euler_xyz_to_matrix(self.euler_angles)
        camera_points = torch.einsum("vij,nj->vni", rotation, self.points3d)
        camera_points = camera_points + self.translations[:, None, :]

        z = camera_points[..., 2]
        z_safe = torch.sign(z) * torch.clamp(z.abs(), min=1e-4)
        u = -self.focal * camera_points[..., 0] / z_safe + self.cx
        v = self.focal * camera_points[..., 1] / z_safe + self.cy
        projected = torch.stack([u, v], dim=-1)
        return projected, camera_points

    def compute_loss(
        self,
        observations: torch.Tensor,
        visibility: torch.Tensor,
        point_reg_weight: float = 1e-4,
        translation_reg_weight: float = 1e-3,
        depth_reg_weight: float = 5e-2,
        focal_reg_weight: float = 1e-4,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        projected, camera_points = self.project()
        residual = projected - observations
        per_point_error = torch.linalg.norm(residual, dim=-1)

        visible_errors = per_point_error[visibility]
        reprojection_loss = torch.sqrt(visible_errors.square() + 1e-6).mean()

        visible_depth = camera_points[..., 2][visibility]
        depth_penalty = F.relu(visible_depth + 1e-3).mean()

        point_reg = self.points3d.square().mean()
        translation_reg = (
            self.translations[:, :2].square().mean()
            + (self.translations[:, 2] + self.depth_init).square().mean()
        )
        focal_reg = ((self.focal - self.focal_init) / self.focal_init).square()

        total_loss = (
            reprojection_loss
            + point_reg_weight * point_reg
            + translation_reg_weight * translation_reg
            + depth_reg_weight * depth_penalty
            + focal_reg_weight * focal_reg
        )

        metrics = {
            "total_loss": float(total_loss.detach().cpu()),
            "reprojection_loss": float(reprojection_loss.detach().cpu()),
            "depth_penalty": float(depth_penalty.detach().cpu()),
            "point_reg": float(point_reg.detach().cpu()),
            "translation_reg": float(translation_reg.detach().cpu()),
            "focal": float(self.focal.detach().cpu()),
            "mean_reprojection_px": float(visible_errors.detach().mean().cpu()),
            "median_reprojection_px": float(visible_errors.detach().median().cpu()),
        }
        return total_loss, metrics, projected


def save_colored_point_cloud_obj(
    path: str | Path,
    points3d: np.ndarray,
    colors: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for point, color in zip(points3d, colors):
            x, y, z = point.tolist()
            r, g, b = color.tolist()
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
