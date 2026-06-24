from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in VALID_IMAGE_SUFFIXES)


def validate_config(config: dict[str, Any], root: Path) -> list[str]:
    issues: list[str] = []

    object_a_images = root / config["object_a"]["image_dir"]
    scene_images = root / config["scene"]["image_dir"]
    object_c_image = root / config["object_c"]["input_image"]

    if count_images(object_a_images) == 0:
        issues.append(f"Missing or empty Object A image dir: {object_a_images}")
    if count_images(scene_images) == 0:
        issues.append(f"Missing or empty scene image dir: {scene_images}")
    if not object_c_image.is_file():
        issues.append(f"Missing Object C input image: {object_c_image}")

    target_asset = config["fusion"]["target_asset"]
    if target_asset not in {"object_a", "object_b", "object_c"}:
        issues.append(f"fusion.target_asset must be one of object_a/object_b/object_c, got: {target_asset}")

    return issues


def build_commands(config: dict[str, Any], config_arg: str) -> dict[str, list[str]]:
    tools = config["tools"]
    object_a = config["object_a"]
    object_b = config["object_b"]
    object_c = config["object_c"]
    scene = config["scene"]
    fusion = config["fusion"]

    object_a_colmap = [
        f'{tools["colmap_bin"]} feature_extractor --database_path {object_a["colmap_workspace"]}/database.db --image_path {object_a["image_dir"]}',
        f'{tools["colmap_bin"]} exhaustive_matcher --database_path {object_a["colmap_workspace"]}/database.db',
        f'{tools["colmap_bin"]} mapper --database_path {object_a["colmap_workspace"]}/database.db --image_path {object_a["image_dir"]} --output_path {object_a["colmap_workspace"]}/sparse',
        f'{tools["gaussian_splatting_train_script"]} -s {object_a["image_dir"]} -m {object_a["gs_output_dir"]}'
    ]

    object_b_text3d = [
        (
            f'{tools["threestudio_launch"]} --config configs/threestudio-text3d.yaml '
            f'system.prompt_processor.prompt="{object_b["prompt"]}" '
            f'trainer.default_root_dir={object_b["workspace"]}'
        )
    ]

    object_c_image3d = [
        (
            f'{tools["zero123_launch"]} --input_image {object_c["input_image"]} '
            f'--workspace {object_c["workspace"]}'
        )
    ]

    scene_reconstruction = [
        f'{tools["colmap_bin"]} feature_extractor --database_path {scene["colmap_workspace"]}/database.db --image_path {scene["image_dir"]}',
        f'{tools["colmap_bin"]} exhaustive_matcher --database_path {scene["colmap_workspace"]}/database.db',
        f'{tools["colmap_bin"]} mapper --database_path {scene["colmap_workspace"]}/database.db --image_path {scene["image_dir"]} --output_path {scene["colmap_workspace"]}/sparse',
        f'{tools["gaussian_splatting_train_script"]} -s {scene["image_dir"]} -m {scene["gs_output_dir"]}'
    ]

    fusion_commands = [
        f"python scripts/hw3_pipeline.py manifest --config {config_arg}",
        (
            f'{tools["blender_bin"]} --python scripts/blender_fusion.py -- '
            f'--manifest {fusion["blender_manifest_path"]}'
        )
    ]

    return {
        "object_a_colmap_3dgs": object_a_colmap,
        "object_b_text_to_3d": object_b_text3d,
        "object_c_image_to_3d": object_c_image3d,
        "scene_colmap_3dgs": scene_reconstruction,
        "fusion": fusion_commands,
    }


def write_manifest(config: dict[str, Any], root: Path) -> Path:
    fusion = config["fusion"]
    manifest_path = root / fusion["blender_manifest_path"]
    ensure_parent(manifest_path)

    payload = {
        "asset_mesh_path": str((root / fusion["asset_mesh_path"]).resolve()),
        "blend_path": str((root / fusion["blend_path"]).resolve()),
        "render_output_dir": str((root / fusion["render_output_dir"]).resolve()),
        "transform": fusion["transform"],
        "camera": fusion["camera"],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


def print_commands(commands: dict[str, list[str]]) -> None:
    for stage, stage_commands in commands.items():
        print(f"[{stage}]")
        for cmd in stage_commands:
            print(cmd)
        print()


def summarize(config: dict[str, Any], root: Path) -> None:
    print(f"Project: {config['project_name']}")
    print(f"Root: {root}")
    print(f"Object A images: {root / config['object_a']['image_dir']}")
    print(f"Object B prompt: {config['object_b']['prompt']}")
    print(f"Object C image: {root / config['object_c']['input_image']}")
    print(f"Scene images: {root / config['scene']['image_dir']}")
    print(f"Fusion target: {config['fusion']['target_asset']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HW3 Task 1 helper pipeline")
    parser.add_argument("action", choices=["validate", "commands", "manifest", "summary"])
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)

    if args.action == "validate":
        issues = validate_config(config, root)
        if issues:
            print("Validation failed:")
            for issue in issues:
                print(f"- {issue}")
            raise SystemExit(1)
        print("Validation passed.")
        return

    if args.action == "commands":
        print_commands(build_commands(config, args.config))
        return

    if args.action == "manifest":
        manifest_path = write_manifest(config, root)
        print(f"Manifest written to: {manifest_path}")
        return

    if args.action == "summary":
        summarize(config, root)


if __name__ == "__main__":
    main()
