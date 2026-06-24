#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(os.environ.get("HW3_TASK1_ROOT", "/root/autodl-tmp/hw3_task1_3dgs_aigc"))
RESULTS = ROOT / "results"
ASSETS = RESULTS / "report_assets"


def _du(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        import subprocess
        return subprocess.check_output(["du", "-sh", str(path)], text=True).split()[0]
    except Exception:
        return "unknown"


def _read_metrics() -> list[dict[str, str]]:
    path = RESULTS / "3dgs_metrics_summary.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_open_image(path: Path, size: tuple[int, int]):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _draw_label(draw, xy: tuple[int, int], text: str) -> None:
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([xy[0], xy[1], xy[0] + max(80, len(text) * 7), xy[1] + 16], fill=(255, 255, 255))
    draw.text((xy[0] + 3, xy[1] + 2), text, fill=(0, 0, 0), font=font)


def _make_3dgs_contact_sheet(variant: str, out_dir: Path) -> Path | None:
    from PIL import Image, ImageDraw
    candidates = sorted(out_dir.glob("train/ours_*/renders")) + sorted(out_dir.glob("test/ours_*/renders"))
    if not candidates:
        return None
    render_dir = candidates[-1]
    gt_dir = render_dir.parent / "gt"
    renders = sorted(render_dir.glob("*.png"))
    if not renders or not gt_dir.exists():
        return None
    if len(renders) <= 4:
        selected = renders
    else:
        idxs = [0, len(renders)//3, (2*len(renders))//3, len(renders)-1]
        selected = [renders[i] for i in idxs]
    cell = (220, 150)
    label_h = 22
    sheet = Image.new("RGB", (cell[0] * len(selected), (cell[1] + label_h) * 2 + 30), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), f"3DGS {variant}: GT (top) vs render (bottom)", fill=(0, 0, 0))
    for col, render_path in enumerate(selected):
        gt_path = gt_dir / render_path.name
        if not gt_path.exists():
            continue
        x = col * cell[0]
        gt = _safe_open_image(gt_path, cell)
        rd = _safe_open_image(render_path, cell)
        sheet.paste(gt, (x, 30))
        sheet.paste(rd, (x, 30 + cell[1] + label_h))
        _draw_label(draw, (x + 4, 32), f"GT {render_path.stem}")
        _draw_label(draw, (x + 4, 32 + cell[1] + label_h), f"Render {render_path.stem}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"3dgs_{variant}_contact.jpg"
    sheet.save(out, quality=92)
    return out


def _make_aigc_contact_sheet() -> Path | None:
    from PIL import Image, ImageDraw
    candidates: list[Path] = []
    for base in [ROOT / "outputs/threestudio", RESULTS / "threestudio_text_to_3d", RESULTS / "zero123_image_to_3d"]:
        if base.exists():
            candidates.extend(sorted(base.rglob("*.png")))
            candidates.extend(sorted(base.rglob("*.jpg")))
    if not candidates:
        return None
    filtered = []
    for path in candidates:
        s = str(path)
        if "it10000-test" in s or "full@" in s or "zero123" in s or "smoke" in s:
            filtered.append(path)
    candidates = filtered or candidates
    if len(candidates) > 12:
        step = max(1, len(candidates) // 12)
        candidates = candidates[::step][:12]
    cell = (180, 180)
    cols = min(4, len(candidates))
    rows = (len(candidates) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell[0], rows * (cell[1] + 22) + 24), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), "AIGC 3D route snapshots: text-to-3D / image-to-3D", fill=(0, 0, 0))
    for i, path in enumerate(candidates):
        r, c = divmod(i, cols)
        x = c * cell[0]
        y = 24 + r * (cell[1] + 22)
        try:
            img = _safe_open_image(path, cell)
        except Exception:
            continue
        sheet.paste(img, (x, y))
        label = path.parent.parent.name if path.parent.name.startswith("it") else path.parent.name
        _draw_label(draw, (x + 4, y + 2), label[:24])
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / "aigc_output_contact.jpg"
    sheet.save(out, quality=92)
    return out


def _index_artifacts() -> list[dict[str, object]]:
    patterns = ["*.ply", "*.obj", "*.glb", "*.mp4", "*.png", "*.jpg", "*.jpeg", "metrics.txt", "cfg_args"]
    rows = []
    for base in [ROOT / "outputs", RESULTS]:
        if not base.exists():
            continue
        for pattern in patterns:
            for path in base.rglob(pattern):
                if path.is_file():
                    rows.append({
                        "path": str(path),
                        "relative_path": str(path.relative_to(ROOT)),
                        "size_bytes": path.stat().st_size,
                        "kind": path.suffix.lower().lstrip(".") or path.name,
                    })
    rows.sort(key=lambda x: x["relative_path"])
    return rows


def _write_metric_markdown(metrics: list[dict[str, str]], contact_sheets: list[Path]) -> Path:
    lines = [
        "# HW3 Task 1 Final Report Artifacts",
        "",
        f"Updated: {datetime.now().astimezone().isoformat()}",
        "",
        "## 3DGS Metrics",
        "",
        "| Variant | Scene | PSNR | SSIM | LPIPS |",
        "|---|---|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(f"| {row.get('variant','')} | {row.get('scene','')} | {row.get('psnr','')} | {row.get('ssim','')} | {row.get('lpips','')} |")
    lines += ["", "## Contact Sheets", ""]
    for path in contact_sheets:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines += [
        "",
        "## Notes",
        "",
        "- The 3DGS route includes multi-scene reconstruction plus garden iteration/resolution ablations.",
        "- The AIGC route includes text-to-3D SDS and image-to-3D Zero123 outputs through threestudio.",
        "- Mesh/checkpoint/render/video artifacts are indexed in `results/task1_final_report_artifacts.json`.",
    ]
    out = RESULTS / "task1_final_report_artifacts.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    metrics = _read_metrics()
    sheets: list[Path] = []
    for variant_dir in sorted((ROOT / "outputs/3dgs").glob("*")):
        if not variant_dir.is_dir():
            continue
        try:
            out = _make_3dgs_contact_sheet(variant_dir.name, variant_dir)
            if out:
                sheets.append(out)
        except Exception as exc:
            print(f"contact sheet failed for {variant_dir.name}: {exc}")
    try:
        aigc = _make_aigc_contact_sheet()
        if aigc:
            sheets.append(aigc)
    except Exception as exc:
        print(f"AIGC contact sheet failed: {exc}")

    artifacts = _index_artifacts()
    md = _write_metric_markdown(metrics, sheets)
    payload = {
        "run_id": "task1_final_report_artifacts",
        "status": "finished",
        "updated_at": datetime.now().astimezone().isoformat(),
        "metrics_rows": metrics,
        "contact_sheets": [str(p) for p in sheets],
        "markdown": str(md),
        "artifact_count": len(artifacts),
        "artifact_index_sample": artifacts[:1000],
        "sizes": {
            "outputs": _du(ROOT / "outputs"),
            "results": _du(ROOT / "results"),
            "data": _du(ROOT / "data"),
            "models": _du(ROOT / "models"),
        },
        "blender_executable": shutil.which("blender"),
    }
    out_json = RESULTS / "task1_final_report_artifacts.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "markdown": str(md), "sheets": len(sheets), "artifacts": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
