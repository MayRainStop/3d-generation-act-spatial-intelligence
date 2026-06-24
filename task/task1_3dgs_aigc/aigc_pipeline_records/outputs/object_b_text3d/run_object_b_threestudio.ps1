$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ThreeStudioRoot = Join-Path $ProjectRoot "external\threestudio"
$LaunchPy = Join-Path $ThreeStudioRoot "launch.py"
$ConfigYaml = Join-Path $ThreeStudioRoot "configs\dreamfusion-sd.yaml"
$OutputDir = Join-Path $ProjectRoot "outputs\object_b_text3d"

python $LaunchPy --config $ConfigYaml --train --gpu 0 name=dreamfusion-sd exp_root_dir="$OutputDir" tag="text3d_banana" system.prompt_processor.prompt="a single ripe Cavendish banana, full body, centered in the frame, isolated on a plain light background, smooth natural yellow peel with a few small brown freckles, gentle elegant curve, realistic fruit proportions, detailed stem and tip, soft diffused studio lighting, subtle shadow, photorealistic, highly detailed surface texture, sharp focus, clean 3D product render, complete object, consistent shape from all views" system.prompt_processor.negative_prompt="blurry, low quality, low resolution, deformed, broken shape, duplicated object, multiple bananas, fruit bunch, plate, basket, hand, table, background clutter, floating fragments, cropped object, cut off, extra parts, unrealistic texture, oversaturated, noisy, distorted geometry"
