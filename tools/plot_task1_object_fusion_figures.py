from __future__ import annotations

import math
import struct
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tasks" / "task1_3dgs_aigc"
REPORT_ASSETS = ROOT / "docs" / "report_assets"
TASK1_ASSETS = ROOT / "results" / "task1-3dgs-aigc" / "report-assets"


def _read_binary_ply_vertices(path: Path, max_points: int = 80000) -> np.ndarray:
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError(f"Only binary little-endian PLY is supported: {path}")

        vertex_count = None
        properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
                in_vertex = True
                continue
            if len(parts) >= 2 and parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            if in_vertex and len(parts) == 3 and parts[0] == "property":
                properties.append((parts[1], parts[2]))

        if vertex_count is None:
            raise ValueError(f"PLY vertex count is missing: {path}")

        dtype_map = {
            "float": ("f4", 4),
            "float32": ("f4", 4),
            "double": ("f8", 8),
            "uchar": ("u1", 1),
            "uint8": ("u1", 1),
            "int": ("i4", 4),
            "uint": ("u4", 4),
        }
        dtype = []
        stride = 0
        for typ, name in properties:
            if typ not in dtype_map:
                raise ValueError(f"Unsupported PLY property type {typ!r} in {path}")
            np_type, width = dtype_map[typ]
            dtype.append((name, "<" + np_type if width > 1 else np_type))
            stride += width

        sample_count = min(vertex_count, max_points)
        raw = f.read(vertex_count * stride)
        vertices = np.frombuffer(raw, dtype=np.dtype(dtype), count=vertex_count)
        if vertex_count > sample_count:
            idx = np.linspace(0, vertex_count - 1, sample_count).astype(np.int64)
            vertices = vertices[idx]
        return vertices


def _scatter_ply(ax, path: Path, title: str, max_points: int = 60000, colored: bool = False) -> None:
    v = _read_binary_ply_vertices(path, max_points=max_points)
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float32)
    xyz -= np.nanmean(xyz, axis=0)
    scale = np.nanmax(np.linalg.norm(xyz, axis=1))
    if scale > 0:
        xyz /= scale

    colors = "#4477aa"
    if colored and {"red", "green", "blue"}.issubset(v.dtype.names or []):
        colors = np.column_stack([v["red"], v["green"], v["blue"]]) / 255.0

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.35, c=colors, alpha=0.85, linewidths=0)
    ax.view_init(elev=18, azim=-58)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    lim = 0.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def _read_obj_vertices(path: Path, max_points: int = 80000) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if not vertices:
        raise ValueError(f"No OBJ vertices found in {path}")

    xyz = np.asarray(vertices, dtype=np.float32)
    if len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points).astype(np.int64)
        xyz = xyz[idx]
    return xyz


def _scatter_obj(ax, path: Path, title: str, color: str, max_points: int = 50000) -> None:
    xyz = _read_obj_vertices(path, max_points=max_points)
    xyz -= np.nanmean(xyz, axis=0)
    scale = np.nanmax(np.linalg.norm(xyz, axis=1))
    if scale > 0:
        xyz /= scale

    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.45, c=color, alpha=0.86, linewidths=0)
    ax.view_init(elev=20, azim=-48)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    lim = 0.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)


def _show_image(ax, path: Path, title: str) -> None:
    ax.imshow(Image.open(path))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _crop_video_frame(video: Path, frame_index: int, box: tuple[int, int, int, int]) -> Image.Image:
    frame = Image.fromarray(iio.imread(video, index=frame_index))
    return frame.crop(box)


def _show_crop(ax, crop: Image.Image, title: str) -> None:
    ax.imshow(crop)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _show_prompt_panel(ax, title: str, prompt: str) -> None:
    ax.set_facecolor("#f5f5f5")
    ax.text(
        0.05,
        0.90,
        prompt,
        va="top",
        ha="left",
        fontsize=9,
        wrap=True,
        transform=ax.transAxes,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_abc_construction_figure() -> Path:
    raw_a = TASK_ROOT / "object_a_reconstruction" / "data" / "object_a_data" / "images" / "frame_0000.jpg"
    final_a_ply = (
        TASK_ROOT
        / "object_a_reconstruction"
        / "outputs"
        / "object_a_3dgs"
        / "point_cloud"
        / "iteration_7000"
        / "point_cloud.ply"
    )
    obj_b = TASK_ROOT / "blender_fusion" / "assets" / "input-assets" / "obj_b.obj"
    obj_c = TASK_ROOT / "blender_fusion" / "assets" / "input-assets" / "obj_c.obj"
    c_input = TASK_ROOT / "aigc_pipeline_records" / "data" / "object_c_image3d" / "inputs" / "object_c_input.png"
    video = TASK_ROOT / "blender_fusion" / "outputs" / "object_fusion_orbit_v5.mp4"

    frame_index = 29
    crop_a = _crop_video_frame(video, frame_index, (455, 440, 790, 720))
    crop_b = _crop_video_frame(video, frame_index, (815, 210, 1135, 610))
    crop_c = _crop_video_frame(video, frame_index, (370, 385, 610, 565))

    fig = plt.figure(figsize=(12.0, 9.4), dpi=180)
    grid = fig.add_gridspec(3, 3, hspace=0.22, wspace=0.10)

    _show_image(fig.add_subplot(grid[0, 0]), raw_a, "A source: phone image")
    _show_prompt_panel(
        fig.add_subplot(grid[0, 1]),
        "B source: text prompt",
        "single ripe pear, yellow-green skin, red blush, speckles, clear pear silhouette",
    )
    _show_image(fig.add_subplot(grid[0, 2]), c_input, "C source: single image")

    ax_a = fig.add_subplot(grid[1, 0], projection="3d")
    _scatter_ply(ax_a, final_a_ply, "A asset: 3DGS cloud", max_points=65000, colored=False)
    ax_b = fig.add_subplot(grid[1, 1], projection="3d")
    _scatter_obj(ax_b, obj_b, "B asset: pear mesh", color="#34b233")
    ax_c = fig.add_subplot(grid[1, 2], projection="3d")
    _scatter_obj(ax_c, obj_c, "C asset: image-to-3D mesh", color="#e34a33")

    _show_crop(fig.add_subplot(grid[2, 0]), crop_a, "A in video frame")
    _show_crop(fig.add_subplot(grid[2, 1]), crop_b, "B in video frame")
    _show_crop(fig.add_subplot(grid[2, 2]), crop_c, "C in video frame")

    fig.suptitle("Object A/B/C construction sources, intermediate assets, and fusion video evidence", fontsize=13)
    out = REPORT_ASSETS / "task1_object_abc_construction.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_object_a_reconstruction_figure() -> Path:
    raw_dir = TASK_ROOT / "object_a_reconstruction" / "data" / "object_a_data" / "images"
    sparse_ply = (
        TASK_ROOT
        / "object_a_reconstruction"
        / "data"
        / "object_a_data"
        / "undistorted"
        / "sparse"
        / "0"
        / "points3D.ply"
    )
    final_ply = (
        TASK_ROOT
        / "object_a_reconstruction"
        / "outputs"
        / "object_a_3dgs"
        / "point_cloud"
        / "iteration_7000"
        / "point_cloud.ply"
    )

    frames = sorted(raw_dir.glob("*.jpg"))
    selected = [frames[i] for i in [0, 14, 29, 43, 58, 72]]

    fig = plt.figure(figsize=(12.0, 7.2), dpi=180)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.12], hspace=0.25, wspace=0.08)

    for col, image_path in enumerate(selected[:4]):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(Image.open(image_path))
        ax.set_title(f"Phone view {col + 1}", fontsize=10)
        ax.axis("off")

    for col, image_path in enumerate(selected[4:]):
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(Image.open(image_path))
        ax.set_title(f"Phone view {col + 5}", fontsize=10)
        ax.axis("off")

    ax_sparse = fig.add_subplot(grid[1, 2], projection="3d")
    _scatter_ply(ax_sparse, sparse_ply, "COLMAP sparse points", max_points=12000, colored=True)

    ax_final = fig.add_subplot(grid[1, 3], projection="3d")
    _scatter_ply(ax_final, final_ply, "3DGS object-A cloud", max_points=70000, colored=False)

    fig.suptitle("Object A phone capture, COLMAP alignment, and 3DGS reconstruction", fontsize=13)
    out = REPORT_ASSETS / "task1_object_a_reconstruction.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_blender_video_contact_sheet() -> Path:
    video = TASK_ROOT / "blender_fusion" / "outputs" / "object_fusion_orbit_v5.mp4"
    props = iio.improps(video)
    meta = iio.immeta(video)
    nframes = props.n_images
    if not nframes or not math.isfinite(float(nframes)):
        nframes = int(round(float(meta.get("fps", 24.0)) * float(meta.get("duration", 6.25))))
    nframes = max(1, int(nframes))
    indices = np.linspace(0, max(0, nframes - 1), 6).astype(int)

    frames = []
    for idx in indices:
        frame = iio.imread(video, index=int(idx))
        frames.append(Image.fromarray(frame).resize((480, 270)))

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2), dpi=180)
    for ax, frame, idx in zip(axes.ravel(), frames, indices):
        ax.imshow(frame)
        ax.set_title(f"Orbit frame {int(idx) + 1}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Blender fusion walkthrough: object A/B/C inserted into the reconstructed counter scene", fontsize=13)
    fig.tight_layout()
    out = REPORT_ASSETS / "task1_blender_fusion_video_frames.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    REPORT_ASSETS.mkdir(parents=True, exist_ok=True)
    TASK1_ASSETS.mkdir(parents=True, exist_ok=True)

    outputs = [
        build_abc_construction_figure(),
        build_object_a_reconstruction_figure(),
        build_blender_video_contact_sheet(),
    ]

    for path in outputs:
        target = TASK1_ASSETS / path.name
        target.write_bytes(path.read_bytes())
        print(path)
        print(target)


if __name__ == "__main__":
    main()
