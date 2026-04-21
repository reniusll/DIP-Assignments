from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COLMAP reconstruction for Assignment 03.")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory containing images/.")
    parser.add_argument("--gpu-index", type=str, default="0", help="COLMAP GPU index, e.g. 0.")
    parser.add_argument(
        "--sift-use-gpu",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use GPU for SIFT feature extraction/matching. Set 0 on headless servers where COLMAP SIFT GPU fails.",
    )
    parser.add_argument("--skip-dense", action="store_true", help="Only run feature extraction, matching, and mapper.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n>>> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()

    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP executable was not found. Please install COLMAP and make sure `colmap` is in PATH.")

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "images"
    colmap_dir = data_dir / "colmap"
    sparse_dir = colmap_dir / "sparse"
    dense_dir = colmap_dir / "dense"
    database_path = colmap_dir / "database.db"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Feature Extraction ===")
    run_command(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.camera_model",
            "PINHOLE",
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.use_gpu",
            str(args.sift_use_gpu),
            "--SiftExtraction.gpu_index",
            args.gpu_index,
        ]
    )

    print("=== Step 2: Feature Matching ===")
    run_command(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.use_gpu",
            str(args.sift_use_gpu),
            "--SiftMatching.gpu_index",
            args.gpu_index,
        ]
    )

    print("=== Step 3: Sparse Reconstruction (Bundle Adjustment) ===")
    run_command(
        [
            "colmap",
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
        ]
    )

    sparse_model_dir = sparse_dir / "0"
    if not sparse_model_dir.exists():
        raise RuntimeError(f"Sparse model was not created at {sparse_model_dir}. Check COLMAP mapper logs.")

    if args.skip_dense:
        print("=== Done: skipped dense reconstruction ===")
        print(f"Sparse result: {sparse_model_dir}")
        return

    print("=== Step 4: Image Undistortion ===")
    run_command(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(sparse_model_dir),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
        ]
    )

    print("=== Step 5: Dense Reconstruction (Patch Match Stereo) ===")
    run_command(
        [
            "colmap",
            "patch_match_stereo",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.gpu_index",
            args.gpu_index,
            "--PatchMatchStereo.geom_consistency",
            "true",
        ]
    )

    print("=== Step 6: Stereo Fusion ===")
    fused_path = dense_dir / "fused.ply"
    run_command(
        [
            "colmap",
            "stereo_fusion",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(fused_path),
        ]
    )

    print("=== Done! ===")
    print(f"Sparse result: {sparse_model_dir}")
    print(f"Dense result:  {fused_path}")


if __name__ == "__main__":
    main()
