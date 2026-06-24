$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ThreeStudioRoot = Join-Path $ProjectRoot "external\threestudio"
$LaunchPy = Join-Path $ThreeStudioRoot "launch.py"
$ConfigYaml = Join-Path $ThreeStudioRoot "configs\stable-zero123.yaml"
$OutputDir = Join-Path $ProjectRoot "outputs\object_c_zero123"
$InputImage = Join-Path $ProjectRoot "data\object_c_image3d\inputs\object_c_input.png"

python $LaunchPy --config $ConfigYaml --train --gpu 0 name=stable-zero123 exp_root_dir="$OutputDir" tag="image3d_single_photo" data.image_path="$InputImage" system.prompt_processor.prompt="image3d_single_photo" system.prompt_processor.negative_prompt="blurry, low quality, deformed, duplicate, floating parts, broken geometry, cropped object, cluttered background"
