from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bundle_adjustment import BundleAdjustmentModel, load_bundle_adjustment_data, save_colored_point_cloud_obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bundle adjustment on the assignment dataset.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/bundle_adjustment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--focal-init", type=float, default=900.0)
    parser.add_argument("--depth-init", type=float, default=2.5)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available on this machine.")
    return torch.device(device_name)


def save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_loss_curve(path: Path, history: list[dict[str, float]]) -> None:
    steps = [row["step"] for row in history]
    total_loss = [row["total_loss"] for row in history]
    reproj = [row["mean_reprojection_px"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, total_loss, color="#1f77b4", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, reproj, color="#d62728", linewidth=2)
    axes[1].set_title("Mean Reprojection Error")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Pixels")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_point_cloud(path: Path, points3d: np.ndarray, colors: np.ndarray) -> None:
    fig = plt.figure(figsize=(12, 4))
    views = [(25, -60), (15, 0), (10, 60)]
    normalized_colors = np.clip(colors, 0.0, 1.0)

    for idx, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.scatter(
            points3d[:, 0],
            points3d[:, 1],
            points3d[:, 2],
            c=normalized_colors,
            s=2,
            marker="o",
            linewidths=0,
            alpha=0.9,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"View {idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_overlays(
    output_dir: Path,
    data_dir: Path,
    observations: np.ndarray,
    projected: np.ndarray,
    visibility: np.ndarray,
    selected_views: list[int],
) -> None:
    overlay_dir = output_dir / "reprojection_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for view_idx in selected_views:
        image_path = data_dir / "images" / f"view_{view_idx:03d}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        vis = visibility[view_idx]
        obs = observations[view_idx][vis]
        pred = projected[view_idx][vis]
        pred = pred[np.isfinite(pred).all(axis=1)]

        for point in obs:
            cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        for point in pred:
            cv2.circle(image, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

        cv2.imwrite(str(overlay_dir / f"view_{view_idx:03d}_overlay.png"), image)


def save_summary(path: Path, metrics: dict[str, float], steps: int) -> None:
    lines = [
        f"Optimization steps: {steps}",
        f"Estimated focal length: {metrics['focal']:.4f}",
        f"Mean reprojection error: {metrics['mean_reprojection_px']:.4f} px",
        f"Median reprojection error: {metrics['median_reprojection_px']:.4f} px",
        f"Final total loss: {metrics['total_loss']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ensure_device(args.device)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_bundle_adjustment_data(data_dir)
    observations = data.observations.to(device)
    visibility = data.visibility.to(device)

    model = BundleAdjustmentModel(
        data=data,
        focal_init=args.focal_init,
        depth_init=args.depth_init,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    history: list[dict[str, float]] = []
    latest_metrics: dict[str, float] | None = None

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, _ = model.compute_loss(observations=observations, visibility=visibility)
        loss.backward()
        optimizer.step()
        scheduler.step()

        metrics["step"] = float(step)
        metrics["lr"] = float(scheduler.get_last_lr()[0])
        history.append(metrics)
        latest_metrics = metrics

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"[step {step:05d}] loss={metrics['total_loss']:.6f} "
                f"reproj={metrics['mean_reprojection_px']:.4f}px focal={metrics['focal']:.4f}"
            )

        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                output_dir / "checkpoint_latest.pth",
            )

    if latest_metrics is None:
        raise RuntimeError("Training did not run.")

    with torch.no_grad():
        _, final_metrics, projected = model.compute_loss(observations=observations, visibility=visibility)
        projected_np = projected.detach().cpu().numpy()
        points3d_np = model.points3d.detach().cpu().numpy()
        rotations_np = model.euler_angles.detach().cpu().numpy()
        translations_np = model.translations.detach().cpu().numpy()
        focal_np = float(model.focal.detach().cpu())

    np.savez(
        output_dir / "bundle_adjustment_result.npz",
        points3d=points3d_np,
        rotations_euler_xyz=rotations_np,
        translations=translations_np,
        focal=focal_np,
        projected=projected_np,
        view_names=np.array(data.view_names),
    )

    save_colored_point_cloud_obj(output_dir / "reconstructed_points.obj", points3d_np, data.colors)
    save_history_csv(output_dir / "history.csv", history)
    plot_loss_curve(output_dir / "loss_curve.png", history)
    plot_point_cloud(output_dir / "point_cloud_views.png", points3d_np, data.colors)
    draw_overlays(
        output_dir=output_dir,
        data_dir=data_dir,
        observations=data.observations.numpy(),
        projected=projected_np,
        visibility=data.visibility.numpy(),
        selected_views=[0, 12, 25, 37, 49],
    )
    save_summary(output_dir / "summary.txt", final_metrics, args.steps)

    print(f"Finished. Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
