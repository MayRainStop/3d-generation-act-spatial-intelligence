import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def deg_list_to_rad(values):
    return [math.radians(v) for v in values]


def main():
    args = parse_args()

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)

    mesh_path = manifest["asset_mesh_path"]
    suffix = Path(mesh_path).suffix.lower()
    if suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif suffix == ".glb":
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {suffix}")

    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not imported_objects:
        raise RuntimeError("No mesh object was imported.")

    target = imported_objects[0]
    transform = manifest["transform"]
    target.location = transform["location"]
    target.rotation_euler = deg_list_to_rad(transform["rotation_euler_deg"])
    target.scale = transform["scale"]

    bpy.ops.object.light_add(type="SUN", location=(5.0, -5.0, 8.0))

    camera_cfg = manifest["camera"]
    bpy.ops.object.camera_add(location=camera_cfg["location"])
    camera = bpy.context.object
    camera.rotation_euler = deg_list_to_rad(camera_cfg["rotation_euler_deg"])
    camera.data.lens = camera_cfg["lens_mm"]
    bpy.context.scene.camera = camera

    render_dir = Path(manifest["render_output_dir"])
    render_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(render_dir / "fusion_preview.png")

    blend_path = Path(manifest["blend_path"])
    blend_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    print(f"Saved blend file to: {blend_path}")
    print(f"Preview output path: {bpy.context.scene.render.filepath}")


if __name__ == "__main__":
    main()
