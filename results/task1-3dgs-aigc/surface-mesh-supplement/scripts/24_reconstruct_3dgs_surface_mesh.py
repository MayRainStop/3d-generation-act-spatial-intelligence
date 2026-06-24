#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
SCENES=['garden_iter30000_r4','bicycle_iter30000_r4','counter_iter30000_r4']
def label(scene): return scene.replace('_iter30000_r4','')
def load_pcd(path):
    p=o3d.io.read_point_cloud(str(path)); p.remove_non_finite_points()
    if len(p.points)==0: raise RuntimeError(f'empty point cloud: {path}')
    if not p.has_colors(): p.colors=o3d.utility.Vector3dVector(np.full((len(p.points),3),0.72))
    return p
def prep(p,target,voxel_frac,normal_frac,seed):
    pts=np.asarray(p.points); b0=pts.min(0); b1=pts.max(0); diag=float(np.linalg.norm(b1-b0))
    if not math.isfinite(diag) or diag<=0: raise RuntimeError('invalid bbox')
    voxel=diag*voxel_frac
    if voxel>0: p=p.voxel_down_sample(voxel)
    if len(p.points)>target:
        idx=np.random.default_rng(seed).choice(len(p.points),target,replace=False)
        p=p.select_by_index(idx.tolist())
    radius=max(diag*normal_frac,1e-6)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius,max_nn=60))
    try: p.orient_normals_consistent_tangent_plane(24)
    except Exception as e:
        print(f'[warn] normal orientation fallback: {e}',flush=True)
        p.orient_normals_towards_camera_location(p.get_center()+np.array([0,-diag,diag*.5]))
    return p, {'bbox_min':b0.tolist(),'bbox_max':b1.tolist(),'bbox_diag':diag,'voxel_size':voxel,'normal_radius':radius,'points':len(p.points)}
def transfer_colors(mesh,p):
    pts=np.asarray(p.points); cols=np.asarray(p.colors); v=np.asarray(mesh.vertices)
    if len(pts) and len(cols) and len(v):
        _,idx=cKDTree(pts).query(v,k=1); mesh.vertex_colors=o3d.utility.Vector3dVector(cols[idx])
def keep_components(mesh,max_components,min_fraction):
    if len(mesh.triangles)==0: return {'components':0,'kept_components':0}
    clusters,ntri,areas=mesh.cluster_connected_triangles(); clusters=np.asarray(clusters); ntri=np.asarray(ntri)
    if len(ntri)==0: return {'components':0,'kept_components':0}
    order=np.argsort(-ntri); min_tri=max(32,int(ntri[order[0]]*min_fraction)); keep=set()
    for c in order[:max_components]:
        if int(ntri[c])>=min_tri: keep.add(int(c))
    mesh.remove_triangles_by_mask(np.array([int(c) not in keep for c in clusters],dtype=bool)); mesh.remove_unreferenced_vertices()
    return {'components':int(len(ntri)),'kept_components':int(len(keep)),'largest_component_triangles':int(ntri[order[0]])}
def reconstruct(scene,in_ply,out,args):
    t=time.time(); raw=load_pcd(in_ply); raw_n=len(raw.points)
    p,pre=prep(raw,args.target_points,args.voxel_fraction,args.normal_radius_fraction,args.seed)
    print(f'[mesh] {scene}: {raw_n} raw -> {len(p.points)} points',flush=True)
    kwargs=dict(depth=args.depth,width=0,scale=args.poisson_scale,linear_fit=False)
    try: mesh,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(p,n_threads=args.threads,**kwargs)
    except TypeError: mesh,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(p,**kwargs)
    dens=np.asarray(dens); thresh=None
    if len(dens):
        thresh=float(np.quantile(dens,args.density_quantile)); mesh.remove_vertices_by_mask(dens<thresh)
    bb=p.get_axis_aligned_bounding_box(); ext=bb.get_extent()
    mesh=mesh.crop(o3d.geometry.AxisAlignedBoundingBox(bb.get_min_bound()-ext*args.crop_margin,bb.get_max_bound()+ext*args.crop_margin))
    comp=keep_components(mesh,args.max_components,args.min_component_fraction)
    mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles(); mesh.remove_duplicated_vertices(); mesh.remove_non_manifold_edges(); mesh.remove_unreferenced_vertices()
    if len(mesh.triangles)>args.max_triangles:
        print(f'[mesh] {scene}: decimate {len(mesh.triangles)} -> {args.max_triangles}',flush=True)
        mesh=mesh.simplify_quadric_decimation(args.max_triangles); mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    transfer_colors(mesh,p); mesh.compute_vertex_normals()
    mdir=out/'meshes'; pdir=out/'reconstruction_point_clouds'; mdir.mkdir(parents=True,exist_ok=True); pdir.mkdir(parents=True,exist_ok=True)
    lab=label(scene); ply=mdir/f'{lab}_surface_mesh_poisson.ply'; obj=mdir/f'{lab}_surface_mesh_poisson.obj'; used=pdir/f'{lab}_surface_reconstruction_points.ply'
    if len(mesh.vertices)==0 or len(mesh.triangles)==0: raise RuntimeError(f'empty mesh for {scene}')
    if not o3d.io.write_triangle_mesh(str(ply),mesh,write_ascii=False,compressed=False): raise RuntimeError(f'failed writing {ply}')
    o3d.io.write_triangle_mesh(str(obj),mesh,write_ascii=False,compressed=False); o3d.io.write_point_cloud(str(used),p,write_ascii=False,compressed=False)
    print(f'[mesh] {scene}: vertices={len(mesh.vertices)} triangles={len(mesh.triangles)} elapsed={time.time()-t:.1f}s',flush=True)
    return {'scene':scene,'label':lab,'input_ply':str(in_ply),'raw_points':raw_n,'preprocess':pre,'mesh_ply':str(ply),'mesh_obj':str(obj),'reconstruction_point_cloud':str(used),'vertices':len(mesh.vertices),'triangles':len(mesh.triangles),'density_threshold':thresh,'component_filter':comp,'elapsed_seconds':time.time()-t}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,default=Path('results/mesh_blender_exports/ply_samples')); ap.add_argument('--output-dir',type=Path,default=Path('results/surface_mesh_exports'))
    ap.add_argument('--target-points',type=int,default=50000); ap.add_argument('--voxel-fraction',type=float,default=1/260); ap.add_argument('--normal-radius-fraction',type=float,default=1/45); ap.add_argument('--depth',type=int,default=8); ap.add_argument('--poisson-scale',type=float,default=1.08); ap.add_argument('--density-quantile',type=float,default=.045); ap.add_argument('--crop-margin',type=float,default=.025); ap.add_argument('--max-components',type=int,default=12); ap.add_argument('--min-component-fraction',type=float,default=.004); ap.add_argument('--max-triangles',type=int,default=220000); ap.add_argument('--threads',type=int,default=8); ap.add_argument('--seed',type=int,default=20260604)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for s in SCENES:
        rows.append(reconstruct(s,args.input_dir/f'{s}_sample120000.ply',args.output_dir,args))
    man={'run_id':'task1_surface_mesh_reconstruction','created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'method':'Open3D Poisson reconstruction from sampled final 3DGS Gaussian centers; normals are estimated on downsampled colored point clouds; low-density vertices and tiny components are pruned.','parameters':{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},'scenes':rows}
    (args.output_dir/'task1_surface_mesh_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[mesh] wrote manifest',args.output_dir/'task1_surface_mesh_manifest.json',flush=True)
if __name__=='__main__': main()
