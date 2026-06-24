#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path("/root/autodl-tmp/hw3_task1_3dgs_aigc")
RESULTS = ROOT / "results" / "3dgs"
OUT_CSV = ROOT / "results" / "3dgs_metrics_summary.csv"
OUT_JSON = ROOT / "results" / "3dgs_metrics_summary.json"


def parse_metrics(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, str] = {}
    patterns = {
        "psnr": r"PSNR\s*[:=]\s*([0-9.]+)",
        "ssim": r"SSIM\s*[:=]\s*([0-9.]+)",
        "lpips": r"LPIPS\s*[:=]\s*([0-9.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            metrics[key] = m.group(1)
    if not metrics:
        # gaussian-splatting metrics.py sometimes prints a JSON-like or table-like blob.
        for key in ["PSNR", "SSIM", "LPIPS"]:
            m = re.search(key + r".*?([0-9]+\.[0-9]+)", text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                metrics[key.lower()] = m.group(1)
    metrics["raw_tail"] = "\\n".join(text.strip().splitlines()[-8:])
    return metrics


rows = []
if RESULTS.exists():
    for metrics_path in sorted(RESULTS.rglob("metrics.txt")):
        variant = metrics_path.parent.name
        row = {
            "variant": variant,
            "scene": variant.split("_iter", 1)[0] if "_iter" in variant else variant,
            "metrics_file": str(metrics_path),
        }
        row.update(parse_metrics(metrics_path))
        rows.append(row)

fieldnames = ["variant", "scene", "psnr", "ssim", "lpips", "metrics_file", "raw_tail"]
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
OUT_JSON.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "rows": len(rows)}, indent=2))
