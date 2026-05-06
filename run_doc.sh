#!/usr/bin/env bash
# Local MkDocs: `uv run --extra docs mkdocs serve --livereload --dirty`.
#   --fast    Skip notebook nbconvert (faster).
#   --stable  No --dirty (full rebuild on each change); fewer search_index / livereload glitches.
#
# If you see search_index.json ERR_CONTENT_LENGTH_MISMATCH or "Network error" in the console,
# try --stable, hard refresh, or disable browser extensions (e.g. Immersive Translate) on 127.0.0.1.
set -euo pipefail
cd "$(dirname "$0")"

fast=0
stable=0
args=()
for a in "$@"; do
  case "$a" in
    --fast) fast=1 ;;
    --stable) stable=1 ;;
    *) args+=("$a") ;;
  esac
done
if [ "$fast" -eq 1 ]; then
  export POWERZOO_DOCS_INCLUDE_NOTEBOOKS=false
fi

mkdir -p docs/zh/examples/notebooks
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete docs/en/examples/notebooks/ docs/zh/examples/notebooks/
else
  find docs/zh/examples/notebooks -maxdepth 1 -name "*.ipynb" -delete
  cp -a docs/en/examples/notebooks/*.ipynb docs/zh/examples/notebooks/
fi
mkdir -p docs/.jupyter_site_build
export JUPYTER_CONFIG_DIR="${PWD}/docs/.jupyter_site_build"
export JUPYTER_DATA_DIR="${PWD}/docs/.jupyter_site_build"

mkdocs_args=(serve --livereload)
if [ "$stable" -eq 0 ]; then
  mkdocs_args+=(--dirty)
fi

# Under `set -u`, empty `"${args[@]}"` errors on some Bash (e.g. macOS 3.2).
if [ "${#args[@]}" -eq 0 ]; then
  exec uv run --extra docs mkdocs "${mkdocs_args[@]}"
else
  exec uv run --extra docs mkdocs "${mkdocs_args[@]}" "${args[@]}"
fi
