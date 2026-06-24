#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/13_download_calvin_dataset.sh <debug|D>" >&2
  exit 2
fi

split="$1"
cd "$(dirname "$0")/.."
mkdir -p data/calvin_raw/datasets logs results

case "$split" in
  debug)
    url="http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip"
    folder="calvin_debug_dataset"
    ;;
  D)
    url="http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
    folder="task_D_D"
    ;;
  *)
    echo "Unknown split: $split. Use debug or D." >&2
    exit 2
    ;;
esac

work_dir="data/calvin_raw/datasets"
zip_path="${work_dir}/${folder}.zip"
status_file="results/download_calvin_${split}.status.json"
log_file="logs/download_calvin_${split}.log"

{
  echo "{"
  echo "  \"run_id\": \"download_calvin_${split}\","
  echo "  \"status\": \"started\","
  echo "  \"start_time\": \"$(date -Iseconds)\","
  echo "  \"url\": \"${url}\","
  echo "  \"target_folder\": \"${work_dir}/${folder}\","
  echo "  \"log_file\": \"${log_file}\""
  echo "}"
} > "$status_file"

exec > >(tee "$log_file") 2>&1

if [ -d "${work_dir}/${folder}" ]; then
  echo "${folder} already exists; skipping download."
else
  echo "Downloading ${url}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      -c \
      -x 8 \
      -s 8 \
      --min-split-size=64M \
      --max-tries=20 \
      --retry-wait=10 \
      --connect-timeout=10 \
      --timeout=30 \
      --file-allocation=none \
      -d "$work_dir" \
      -o "$(basename "$zip_path")" \
      "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c --tries=20 --waitretry=10 --timeout=30 -O "$zip_path" "$url"
  else
    curl -L --fail --retry 20 --retry-delay 10 --continue-at - "$url" -o "$zip_path"
  fi
  echo "Unzipping ${zip_path}"
  unzip -q "$zip_path" -d "$work_dir"
  rm -f "$zip_path"
fi

du -sh "${work_dir}/${folder}" || true

{
  echo "{"
  echo "  \"run_id\": \"download_calvin_${split}\","
  echo "  \"status\": \"finished\","
  echo "  \"exit_code\": 0,"
  echo "  \"end_time\": \"$(date -Iseconds)\","
  echo "  \"target_folder\": \"${work_dir}/${folder}\","
  echo "  \"log_file\": \"${log_file}\""
  echo "}"
} > "$status_file"
