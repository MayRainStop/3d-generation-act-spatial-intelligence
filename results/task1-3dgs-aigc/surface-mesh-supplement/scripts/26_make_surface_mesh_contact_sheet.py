#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
SCENES=['garden','bicycle','counter']; VIEWS=['front','side','iso']
def font(size):
    p=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(p),size) if p.exists() else ImageFont.load_default()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--render-dir',type=Path,default=Path('results/surface_mesh_exports/blender_renders')); ap.add_argument('--out',type=Path,default=Path('results/report_assets/3dgs_surface_mesh_contact.jpg')); ap.add_argument('--mirror-out',type=Path,default=Path('results/surface_mesh_exports/3dgs_surface_mesh_contact.jpg')); a=ap.parse_args()
    tile_w,tile_h,label_h,margin=520,352,40,18; sheet=Image.new('RGB',(margin+3*(tile_w+margin),margin+3*(tile_h+label_h+margin)),(246,247,248)); draw=ImageDraw.Draw(sheet); f=font(20)
    for r,s in enumerate(SCENES):
        for c,v in enumerate(VIEWS):
            p=a.render_dir/s/f'{s}_surface_mesh_{v}.png'; img=Image.open(p).convert('RGB'); img.thumbnail((tile_w,tile_h),Image.LANCZOS)
            x=margin+c*(tile_w+margin); y=margin+r*(tile_h+label_h+margin); tile=Image.new('RGB',(tile_w,tile_h),(238,240,242)); tile.paste(img,((tile_w-img.width)//2,(tile_h-img.height)//2)); sheet.paste(tile,(x,y)); draw.rectangle((x,y+tile_h,x+tile_w,y+tile_h+label_h),fill=(31,35,41)); draw.text((x+14,y+tile_h+8),f'{s} - {v}',fill=(255,255,255),font=f)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.mirror_out.parent.mkdir(parents=True,exist_ok=True); sheet.save(a.out,quality=92); sheet.save(a.mirror_out,quality=92)
    (a.mirror_out.parent/'task1_surface_mesh_contact_manifest.json').write_text(json.dumps({'run_id':'task1_surface_mesh_contact_sheet','contact_sheet':str(a.out),'mirror_contact_sheet':str(a.mirror_out),'scenes':SCENES,'views':VIEWS},indent=2,ensure_ascii=False),encoding='utf-8')
    print('[contact] wrote',a.out)
if __name__=='__main__': main()
