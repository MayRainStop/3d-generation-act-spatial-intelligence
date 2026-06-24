#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import bpy
from mathutils import Vector
SCENES=['garden','bicycle','counter']; VIEWS={'front':(0,-1.8,.35),'side':(1.8,-.25,.35),'iso':(1.35,-1.35,.85)}
def clear():
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
    for db in (bpy.data.meshes,bpy.data.materials,bpy.data.images,bpy.data.cameras,bpy.data.lights):
        for x in list(db): db.remove(x)
def material():
    m=bpy.data.materials.new('vertex_color_surface'); m.use_nodes=True; bsdf=m.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        attr=m.node_tree.nodes.new(type='ShaderNodeAttribute'); attr.attribute_name='Col'; m.node_tree.links.new(attr.outputs['Color'],bsdf.inputs['Base Color']); bsdf.inputs['Roughness'].default_value=.72
    return m
def look_at(obj,target):
    d=target-obj.location; obj.rotation_euler=d.to_track_quat('-Z','Y').to_euler()
def import_obj(path):
    bpy.ops.import_mesh.ply(filepath=str(path)); obj=bpy.context.object; obj.name=path.stem; obj.data.materials.append(material()); bpy.ops.object.shade_smooth(); return obj
def center(obj):
    bpy.context.view_layer.update(); pts=[obj.matrix_world@Vector(c) for c in obj.bound_box]
    mn=Vector((min(p.x for p in pts),min(p.y for p in pts),min(p.z for p in pts))); mx=Vector((max(p.x for p in pts),max(p.y for p in pts),max(p.z for p in pts)))
    obj.location-=(mn+mx)*.5; bpy.context.view_layer.update(); pts=[obj.matrix_world@Vector(c) for c in obj.bound_box]
    mn=Vector((min(p.x for p in pts),min(p.y for p in pts),min(p.z for p in pts))); mx=Vector((max(p.x for p in pts),max(p.y for p in pts),max(p.z for p in pts)))
    return max(float((mx-mn).x),float((mx-mn).y),float((mx-mn).z),1e-4)
def safe_set_color_management(scene):
    try: scene.view_settings.view_transform='Standard'
    except Exception as e: print(f'[render] view_transform fallback kept: {e}')
    try: scene.view_settings.look='None'
    except Exception as e: print(f'[render] look fallback kept: {e}')
def setup(rx,ry):
    s=bpy.context.scene; s.render.engine='BLENDER_EEVEE'; s.eevee.use_gtao=True; s.eevee.gtao_distance=3; s.eevee.gtao_factor=1.5; s.render.resolution_x=rx; s.render.resolution_y=ry; safe_set_color_management(s)
    if s.world is not None: s.world.color=(.96,.97,.98)
def render_scene(scene,mesh,out,rx,ry):
    clear(); setup(rx,ry); obj=import_obj(mesh); extent=center(obj); scale=extent*1.25
    bpy.ops.object.light_add(type='AREA',location=(0,-scale*1.2,scale*1.8)); bpy.context.object.data.energy=550; bpy.context.object.data.size=max(scale*1.8,1)
    bpy.ops.object.light_add(type='SUN',location=(scale,-scale,scale)); bpy.context.object.data.energy=1.25
    bpy.ops.object.camera_add(location=(0,-scale*2,scale*.5)); cam=bpy.context.object; cam.data.type='ORTHO'; bpy.context.scene.camera=cam
    od=out/scene; od.mkdir(parents=True,exist_ok=True); files=[]
    for v,d in VIEWS.items():
        cam.location=Vector(d)*scale; look_at(cam,Vector((0,0,0))); cam.data.ortho_scale=max(extent*(1.22 if v=='iso' else 1.16),1e-3)
        p=od/f'{scene}_surface_mesh_{v}.png'; bpy.context.scene.render.filepath=str(p); bpy.ops.render.render(write_still=True); files.append(str(p))
    return files
def main():
    if '--' in sys.argv: sys.argv=[sys.argv[0]]+sys.argv[sys.argv.index('--')+1:]
    ap=argparse.ArgumentParser(); ap.add_argument('--mesh-dir',type=Path,default=Path('results/surface_mesh_exports/meshes')); ap.add_argument('--out-dir',type=Path,default=Path('results/surface_mesh_exports/blender_renders')); ap.add_argument('--resolution-x',type=int,default=1400); ap.add_argument('--resolution-y',type=int,default=950); a=ap.parse_args()
    rendered=[]
    for s in SCENES: rendered+=render_scene(s,a.mesh_dir/f'{s}_surface_mesh_poisson.ply',a.out_dir,a.resolution_x,a.resolution_y)
    missing=[p for p in rendered if not Path(p).exists() or Path(p).stat().st_size==0]
    if missing: raise RuntimeError('missing rendered files: '+', '.join(missing))
    (a.out_dir.parent/'task1_surface_mesh_blender_render_manifest.json').write_text(json.dumps({'run_id':'task1_surface_mesh_blender_render','renderer':'Blender 3.0.1 EEVEE orthographic multi-view render','rendered_files':rendered},indent=2,ensure_ascii=False),encoding='utf-8')
    print('[render] wrote surface mesh render manifest')
if __name__=='__main__': main()
