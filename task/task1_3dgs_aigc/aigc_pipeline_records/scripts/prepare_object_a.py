from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(root: Path, config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.is_absolute():
        return config_path
    return (root / config_path).resolve()


def collect_images(image_dir: Path) -> list[Path]:
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def inspect_images(image_paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    issues: list[str] = []

    for path in image_paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
                records.append(
                    {
                        "filename": path.name,
                        "suffix": path.suffix.lower(),
                        "width": width,
                        "height": height,
                        "mode": img.mode,
                    }
                )
        except Exception as exc:
            issues.append(f"Failed to read image {path.name}: {exc}")

    return records, issues


def build_summary(config: dict[str, Any], records: list[dict[str, Any]], issues: list[str]) -> dict[str, Any]:
    widths = [item["width"] for item in records]
    heights = [item["height"] for item in records]
    suffix_counts = Counter(item["suffix"] for item in records)
    mode_counts = Counter(item["mode"] for item in records)
    unique_sizes = sorted({(item["width"], item["height"]) for item in records})

    object_a = config["object_a"]
    warnings: list[str] = list(issues)

    if len(records) < object_a["min_recommended_images"]:
        warnings.append(
            f"Only {len(records)} images found; recommended at least {object_a['min_recommended_images']}."
        )
    if len(unique_sizes) > 1:
        warnings.append("Mixed image resolutions detected. Consistent resolution is preferred for easier debugging.")
    if len(suffix_counts) > 1:
        warnings.append("Mixed image formats detected. Prefer a single format such as JPG.")

    return {
        "object_name": object_a["name"],
        "image_count": len(records),
        "min_recommended_images": object_a["min_recommended_images"],
        "width_range": [min(widths), max(widths)] if widths else None,
        "height_range": [min(heights), max(heights)] if heights else None,
        "unique_sizes": unique_sizes,
        "format_counts": dict(suffix_counts),
        "mode_counts": dict(mode_counts),
        "warnings": warnings,
        "images": records,
    }


def export_prepared_images(image_paths: list[Path], prepared_dir: Path) -> list[dict[str, str]]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    for index, src in enumerate(image_paths, start=1):
        dst_name = f"frame_{index:04d}{src.suffix.lower()}"
        dst = prepared_dir / dst_name
        shutil.copy2(src, dst)
        manifest.append({"source": str(src), "prepared": str(dst)})

    return manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_colmap_script(config: dict[str, Any], root: Path) -> Path:
    object_a = config["object_a"]
    tools = config["tools"]

    script_path = root / "outputs" / "object_a_3dgs" / "run_object_a_colmap.ps1"
    script_path.parent.mkdir(parents=True, exist_ok=True)

    prepared_image_dir = str((root / object_a["prepared_image_dir"]).resolve())
    colmap_workspace = str((root / object_a["colmap_workspace"]).resolve())
    gs_output_dir = str((root / object_a["gs_output_dir"]).resolve())
    gs_root = str((root / tools["gaussian_splatting_root"]).resolve())

    content = f"""$ErrorActionPreference = 'Stop'
$workspace = "{colmap_workspace}"
$images = "{prepared_image_dir}"
$modelOut = "{gs_output_dir}"
$gsRoot = "{gs_root}"

New-Item -ItemType Directory -Force -Path $workspace | Out-Null
New-Item -ItemType Directory -Force -Path "$workspace\\sparse" | Out-Null
New-Item -ItemType Directory -Force -Path $modelOut | Out-Null

{tools["colmap_bin"]} feature_extractor --database_path "$workspace/database.db" --image_path $images
{tools["colmap_bin"]} exhaustive_matcher --database_path "$workspace/database.db"
{tools["colmap_bin"]} mapper --database_path "$workspace/database.db" --image_path $images --output_path "$workspace/sparse"
Set-Location $gsRoot
{tools["gaussian_splatting_train_script"]} -s $images -m $modelOut
"""

    script_path.write_text(content, encoding="utf-8")
    return script_path


def print_capture_tips() -> None:
    print("Capture tips for Object A:")
    print("- Use 24 to 60 photos around the object.")
    print("- Keep the object fixed and move the camera around it.")
    print("- Prefer uniform lighting and avoid motion blur.")
    print("- Avoid reflective, transparent, or textureless objects.")
    print("- Let the object occupy most of the frame while staying fully visible.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Object A images for COLMAP + 3DGS")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--skip-export", action="store_true", help="Do not copy and rename images")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(resolve_config_path(root, args.config))

    object_a = config["object_a"]
    image_dir = root / object_a["image_dir"]
    prepared_dir = root / object_a["prepared_image_dir"]
    summary_path = root / object_a["summary_path"]

    image_paths = collect_images(image_dir) if image_dir.is_dir() else []
    records, issues = inspect_images(image_paths)
    summary = build_summary(config, records, issues)

    if not args.skip_export and image_paths:
        export_manifest = export_prepared_images(image_paths, prepared_dir)
        summary["prepared_images"] = export_manifest
        summary["prepared_image_dir"] = str(prepared_dir.resolve())

    colmap_script = write_colmap_script(config, root)
    summary["generated_colmap_script"] = str(colmap_script.resolve())

    write_json(summary_path, summary)

    print(f"Object A summary written to: {summary_path}")
    print(f"Generated COLMAP+3DGS script: {colmap_script}")
    print(f"Detected image count: {summary['image_count']}")
    if summary["warnings"]:
        print("Warnings:")
        for item in summary["warnings"]:
            print(f"- {item}")
    print_capture_tips()


if __name__ == "__main__":
    main()
