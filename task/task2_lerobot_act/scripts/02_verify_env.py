from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def run_text(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return "missing"
    return (completed.stdout + completed.stderr).strip()


def main() -> int:
    required_modules = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "matplotlib",
        "yaml",
        "huggingface_hub",
        "datasets",
        "lerobot",
    ]
    report: dict[str, object] = {
        "python": sys.version,
        "executables": {
            "python": sys.executable,
            "lerobot-train": shutil.which("lerobot-train"),
            "nvidia-smi": shutil.which("nvidia-smi"),
        },
        "versions": {},
        "cuda": {},
        "lerobot_train_help_head": "",
    }

    missing: list[str] = []
    for name in required_modules:
        try:
            report["versions"][name] = module_version(name)
        except Exception as exc:
            report["versions"][name] = f"ERROR: {exc}"
            missing.append(name)

    try:
        import torch

        report["cuda"] = {
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        report["cuda"] = {"ERROR": str(exc)}

    help_text = run_text(["lerobot-train", "--help"])
    report["lerobot_train_help_head"] = "\n".join(help_text.splitlines()[:80])

    out_path = RESULTS / "env_verify.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if missing:
        print(f"Missing modules: {missing}", file=sys.stderr)
        return 1
    if not report["executables"]["lerobot-train"]:
        print("Missing lerobot-train executable", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
