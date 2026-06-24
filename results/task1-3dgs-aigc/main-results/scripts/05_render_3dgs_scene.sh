#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/hw3_task1_3dgs_aigc
source scripts/_common.sh
SCENE="${1:-garden}"
OUT_DIR="${2:-$HW3_TASK1_ROOT/outputs/3dgs/${SCENE}_iter30000_r4}"
VARIANT="$(basename "$OUT_DIR")"
RUN_ID="render_3dgs_${VARIANT}"
write_status "$RUN_ID" running null "rendering $VARIANT"
ENV_PREFIX="$HW3_TASK1_ROOT/envs/3dgs"
GS_DIR="$HW3_TASK1_ROOT/third_party/gaussian-splatting"
RESULT_DIR="$HW3_TASK1_ROOT/results/3dgs/$VARIANT"
activate_conda "$ENV_PREFIX"
export CUDA_HOME="${CUDA_HOME:-$ENV_PREFIX}"
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$GS_DIR/submodules/simple-knn:$GS_DIR/submodules/diff-gaussian-rasterization:${PYTHONPATH:-}"
mkdir -p "$RESULT_DIR"
cd "$GS_DIR"
if [ -f "$OUT_DIR/results.json" ] && [ -s "$RESULT_DIR/metrics.txt" ]; then
  echo "[$(now_iso)] Reusing existing render outputs and metrics for $VARIANT."
else
  python render.py -m "$OUT_DIR" || true
  python metrics.py -m "$OUT_DIR" > "$RESULT_DIR/metrics.txt" 2>&1 || true
fi
cp -f "$OUT_DIR/cfg_args" "$RESULT_DIR/cfg_args" 2>/dev/null || true
while IFS= read -r f; do
  rel="${f#$OUT_DIR/}"
  mkdir -p "$RESULT_DIR/files/$(dirname "$rel")"
  cp -f "$f" "$RESULT_DIR/files/$rel" 2>/dev/null || true
done < <(find "$OUT_DIR" -maxdepth 5 -type f \( -name '*.json' -o -name '*.txt' -o -name '*.png' -o -name '*.jpg' \) | head -n 120)
/root/miniconda3/bin/python - <<PY
import json, pathlib, datetime, os, subprocess
root=pathlib.Path(os.environ['HW3_TASK1_ROOT'])
scene='$SCENE'
variant='$VARIANT'
out=pathlib.Path('$OUT_DIR')
res=root/'results'/'3dgs'/variant
payload={
    'run_id':'render_3dgs_'+variant,
    'status':'finished',
    'exit_code':0,
    'updated_at':datetime.datetime.now().astimezone().isoformat(),
    'scene':scene,
    'variant':variant,
    'model_dir':str(out),
    'result_dir':str(res),
}
try:
    payload['model_size']=subprocess.check_output(['du','-sh',str(out)], text=True).split()[0]
except Exception:
    pass
with open(root/'status'/f'render_3dgs_{variant}.json','w',encoding='utf-8') as f:
    json.dump(payload,f,indent=2)
PY
