#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
SCENE="${1:-garden}"
if [ "$SCENE" = "all" ]; then
  for s in garden bicycle counter; do
    bash scripts/03_download_mipnerf360.sh "$s"
  done
  exit 0
fi
case "$SCENE" in
  garden|bicycle|counter) ;;
  *) echo "Unsupported scene: $SCENE" >&2; exit 2 ;;
esac
RUN_ID="download_mipnerf360_${SCENE}"
write_status "$RUN_ID" running null "downloading raw Mip-NeRF360 360_v2.zip for $SCENE"
DOWNLOAD_DIR="$HW3_TASK1_ROOT/data/downloads"
DATA_DIR="$HW3_TASK1_ROOT/data/mipnerf360"
ZIP="$DOWNLOAD_DIR/360_v2.zip"
URL="http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR"
# Remove known wrong nerfbaselines result extraction from an earlier URL mix-up.
rm -rf "$DATA_DIR/checkpoint" "$DATA_DIR/predictions" "$DATA_DIR/tensorboard" "$DATA_DIR/results.json" 2>/dev/null || true
rm -f "$DOWNLOAD_DIR/${SCENE}.zip" 2>/dev/null || true
if [ -d "$DATA_DIR/$SCENE/images" ] && [ -d "$DATA_DIR/$SCENE/sparse" ]; then
  write_status "$RUN_ID" finished 0 "$SCENE already extracted"
  exit 0
fi
log_section "download raw Mip-NeRF360 360_v2.zip for $SCENE"
if [ ! -f "$ZIP" ]; then
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 --min-split-size=64M --max-tries=20 --retry-wait=10 --connect-timeout=15 --timeout=60 --file-allocation=none -d "$DOWNLOAD_DIR" -o "360_v2.zip" "$URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$ZIP" "$URL"
  else
    curl -L -C - -o "$ZIP" "$URL"
  fi
else
  echo "Using existing $ZIP"
fi
log_section "extract $SCENE from 360_v2.zip"
mkdir -p "$DATA_DIR"
if unzip -l "$ZIP" "$SCENE/*" >/dev/null 2>&1; then
  unzip -q -o "$ZIP" "$SCENE/*" -d "$DATA_DIR"
elif unzip -l "$ZIP" "360_v2/$SCENE/*" >/dev/null 2>&1; then
  unzip -q -o "$ZIP" "360_v2/$SCENE/*" -d "$DATA_DIR"
  if [ -d "$DATA_DIR/360_v2/$SCENE" ]; then
    rm -rf "$DATA_DIR/$SCENE"
    mv "$DATA_DIR/360_v2/$SCENE" "$DATA_DIR/$SCENE"
    rmdir "$DATA_DIR/360_v2" 2>/dev/null || true
  fi
else
  write_status "$RUN_ID" failed 3 "scene $SCENE not found inside 360_v2.zip"
  exit 3
fi
if [ ! -d "$DATA_DIR/$SCENE/images" ] || [ ! -d "$DATA_DIR/$SCENE/sparse" ]; then
  write_status "$RUN_ID" failed 1 "raw scene images/sparse missing after extraction"
  exit 1
fi
# Keep the archive until all planned scenes exist, then remove it as cleanup.
if [ -d "$DATA_DIR/garden/images" ] && [ -d "$DATA_DIR/bicycle/images" ] && [ -d "$DATA_DIR/counter/images" ]; then
  rm -f "$ZIP"
  archive_removed=true
else
  archive_removed=false
fi
/root/miniconda3/bin/python - <<PY
import json, os, pathlib, datetime, subprocess
root=pathlib.Path(os.environ['HW3_TASK1_ROOT'])
scene='$SCENE'
scene_dir=root/'data'/'mipnerf360'/scene
size=subprocess.check_output(['du','-sh',str(scene_dir)], text=True).split()[0]
payload={
 'run_id':'download_mipnerf360_'+scene,
 'status':'finished',
 'exit_code':0,
 'updated_at':datetime.datetime.now().astimezone().isoformat(),
 'scene':scene,
 'scene_dir':str(scene_dir),
 'size':size,
 'archive_removed': '$archive_removed' == 'true',
 'source_url':'$URL'
}
with open(root/'status'/f'download_mipnerf360_{scene}.json','w',encoding='utf-8') as f:
 json.dump(payload,f,indent=2)
PY
