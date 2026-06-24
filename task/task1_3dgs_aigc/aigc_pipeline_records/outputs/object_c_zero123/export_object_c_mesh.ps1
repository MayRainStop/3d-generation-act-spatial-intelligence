$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ThreeStudioRoot = Join-Path $ProjectRoot "external\threestudio"
$LaunchPy = Join-Path $ThreeStudioRoot "launch.py"

# Replace the parsed.yaml and checkpoint path after training finishes.
python $LaunchPy --config "PATH_TO_PARSED_YAML" --export --gpu 0 resume="PATH_TO_LAST_CKPT" system.exporter_type=mesh-exporter system.exporter.fmt=obj
