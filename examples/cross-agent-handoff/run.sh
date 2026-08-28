#!/bin/sh
set -eu

example_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output_dir=${1:-"$PWD/memhandoff-demo"}

if [ -e "$output_dir" ]; then
    echo "Refusing to overwrite existing path: $output_dir" >&2
    echo "Pass a new output directory as the first argument." >&2
    exit 2
fi

mkdir -p "$output_dir/archive"

echo '$ open-context handoff agent-a.jsonl --store demo/archive --out demo/export.ctx'
open-context handoff "$example_dir/agent-a.jsonl" \
    --store "$output_dir/archive" \
    --out "$output_dir/export.ctx" \
    --title "Customer export writer"

echo
echo '$ open-context validate demo/export.ctx --archive demo/archive'
open-context validate "$output_dir/export.ctx" --archive "$output_dir/archive"

echo
echo '$ open-context compile demo/export.ctx --target generic --budget 800 --task "Continue the export work."'
open-context compile "$output_dir/export.ctx" \
    --target generic \
    --budget 800 \
    --task "Continue Agent A's export work. State the next step and preserve every constraint." \
    > "$output_dir/agent-b-context.json"

echo
echo "Agent B's context is ready: $output_dir/agent-b-context.json"
echo "Paste its \"text\" value into another agent and ask it to continue."
