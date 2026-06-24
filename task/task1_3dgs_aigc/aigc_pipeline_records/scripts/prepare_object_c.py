from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_config_path(root: Path, config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.is_absolute():
        return config_path
    return (root / config_path).resolve()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Object C threestudio Zero123 commands")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--gpu", default="0", help="GPU id passed to threestudio")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = load_config(resolve_config_path(root, args.config))

    object_c = config["object_c"]
    tools = config["tools"]

    threestudio_root = (root / tools["threestudio_root"]).resolve()
    launch_py = threestudio_root / "launch.py"
    config_yaml = (threestudio_root / object_c["threestudio_config"]).resolve()
    input_image = (root / object_c["input_image"]).resolve()
    workspace = (root / object_c["workspace"]).resolve()
    exp_root_dir = workspace.parent.as_posix()

    script_path = root / "outputs" / "object_c_zero123" / "run_object_c_zero123.ps1"
    export_script_path = root / "outputs" / "object_c_zero123" / "export_object_c_mesh.ps1"
    summary_path = root / "outputs" / "object_c_zero123" / "object_c_summary.json"

    train_command = (
        f'python "{launch_py}" --config "{config_yaml}" --train --gpu {args.gpu} '
        f'name=stable-zero123 exp_root_dir="{exp_root_dir}" tag="{object_c["name"]}" '
        f'data.image_path="{input_image}" '
        f'system.prompt_processor.prompt="{object_c["name"]}" '
        f'system.prompt_processor.negative_prompt="{object_c["negative_prompt"]}"'
    )

    script_content = f"""$ErrorActionPreference = 'Stop'
Set-Location "{threestudio_root}"

{train_command}
"""

    export_content = f"""$ErrorActionPreference = 'Stop'
Set-Location "{threestudio_root}"

# Replace the parsed.yaml and checkpoint path after training finishes.
python "{launch_py}" --config "PATH_TO_PARSED_YAML" --export --gpu {args.gpu} resume="PATH_TO_LAST_CKPT" system.exporter_type=mesh-exporter system.exporter.fmt=obj
"""

    write_text(script_path, script_content)
    write_text(export_script_path, export_content)

    summary = {
        "object_name": object_c["name"],
        "threestudio_root": str(threestudio_root),
        "train_script": str(script_path.resolve()),
        "export_script": str(export_script_path.resolve()),
        "train_config": str(config_yaml),
        "workspace": str(workspace),
        "gpu": args.gpu,
        "input_image": str(input_image),
        "negative_prompt": object_c["negative_prompt"],
        "train_command": train_command,
        "notes": [
            "Run the training script first.",
            "After training, find parsed.yaml and ckpts/last.ckpt under the generated trial directory.",
            "Edit export_object_c_mesh.ps1 with those two real paths, then run it to export an OBJ mesh."
        ],
    }
    write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Object C training script written to: {script_path}")
    print(f"Object C export script written to: {export_script_path}")
    print(f"Summary written to: {summary_path}")
    print("Training command:")
    print(train_command)


if __name__ == "__main__":
    main()
