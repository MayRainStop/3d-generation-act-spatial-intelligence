#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/calvin_raw logs results

status_file="results/setup_calvin_eval.status.json"
log_file="logs/setup_calvin_eval.log"
root_dir="data/calvin_raw/calvin"
tmp_dir="data/calvin_raw/.setup_tmp"

{
  echo "{"
  echo "  \"run_id\": \"setup_calvin_eval\","
  echo "  \"status\": \"started\","
  echo "  \"start_time\": \"$(date -Iseconds)\","
  echo "  \"log_file\": \"${log_file}\""
  echo "}"
} > "$status_file"

exec > >(tee "$log_file") 2>&1

echo "Setting up official CALVIN evaluator code at ${root_dir}"
echo "This script keeps extracted code and removes downloaded tarballs."

if [ ! -d "$root_dir/calvin_models/calvin_agent" ] || [ ! -d "$root_dir/calvin_env/calvin_env" ]; then
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"

  echo "Downloading CALVIN main repository tarball..."
  curl -L --fail --retry 5 --connect-timeout 20 \
    https://codeload.github.com/mees/calvin/tar.gz/refs/heads/main \
    -o "$tmp_dir/calvin-main.tar.gz"

  echo "Downloading calvin_env submodule tarball..."
  curl -L --fail --retry 5 --connect-timeout 20 \
    https://codeload.github.com/mees/calvin_env/tar.gz/refs/heads/main \
    -o "$tmp_dir/calvin_env-main.tar.gz"

  echo "Extracting repositories..."
  rm -rf "$tmp_dir/main" "$tmp_dir/env" "$root_dir"
  mkdir -p "$tmp_dir/main" "$tmp_dir/env"
  tar -xzf "$tmp_dir/calvin-main.tar.gz" -C "$tmp_dir/main" --strip-components=1
  tar -xzf "$tmp_dir/calvin_env-main.tar.gz" -C "$tmp_dir/env" --strip-components=1

  mkdir -p "$(dirname "$root_dir")"
  mv "$tmp_dir/main" "$root_dir"
  rm -rf "$root_dir/calvin_env"
  mkdir -p "$root_dir/calvin_env"
  shopt -s dotglob
  mv "$tmp_dir/env"/* "$root_dir/calvin_env/"
  shopt -u dotglob
  rm -rf "$tmp_dir"
else
  echo "CALVIN code already exists; skipping download."
fi

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

echo "Installing minimal evaluator dependencies without changing torch/torchvision..."
python -m pip install \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  hydra-colorlog \
  "gym==0.26.2" \
  numpy-quaternion

echo "Verifying imports..."
PYTHONPATH="$PWD/$root_dir/calvin_models:$PWD/$root_dir/calvin_env:${PYTHONPATH:-}" python - <<'PY'
import importlib.util

mods = [
    "calvin_agent.evaluation.multistep_sequences",
    "calvin_agent.evaluation.utils",
    "calvin_env.envs.play_table_env",
    "hydra",
    "omegaconf",
    "gym",
    "quaternion",
]
for mod in mods:
    spec = importlib.util.find_spec(mod)
    print(f"{mod}: {bool(spec)}")
    if spec is None:
        raise SystemExit(f"Missing import: {mod}")
PY

{
  echo "{"
  echo "  \"run_id\": \"setup_calvin_eval\","
  echo "  \"status\": \"finished\","
  echo "  \"exit_code\": 0,"
  echo "  \"end_time\": \"$(date -Iseconds)\","
  echo "  \"calvin_root\": \"${root_dir}\","
  echo "  \"log_file\": \"${log_file}\""
  echo "}"
} > "$status_file"

echo "CALVIN evaluator setup complete."
