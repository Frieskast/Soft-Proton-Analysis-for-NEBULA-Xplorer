#!/usr/bin/env bash
set -euo pipefail

APP="./thesis"                # path to your executable (adjust)
PHYS_HIGH=( "-phys" "option4" )
PHYS_LOW=(  "-phys" "ss" )

# Optional args: first = high macros dir, second = low macros dir
MACRO_DIR_HIGH="${1:-./macros/high}"
MACRO_DIR_LOW="${2:-./macros/low}"

OUT="output.txt"

# sanity checks
if [ ! -x "$APP" ]; then
  echo "ERROR: $APP not found or not executable" >&2
  exit 1
fi
if [ ! -d "$MACRO_DIR_HIGH" ]; then
  echo "WARNING: high macro dir '$MACRO_DIR_HIGH' not found; skipping high runs" >&2
fi
if [ ! -d "$MACRO_DIR_LOW" ]; then
  echo "WARNING: low macro dir '$MACRO_DIR_LOW' not found; skipping low runs" >&2
fi

# start fresh output file
: > "$OUT"
echo "Run started: $(date -u)" >> "$OUT"
echo "Physics high: ${PHYS_HIGH[*]}" >> "$OUT"
echo "Physics low : ${PHYS_LOW[*]}"  >> "$OUT"
echo "High macros dir: $MACRO_DIR_HIGH" >> "$OUT"
echo "Low  macros dir: $MACRO_DIR_LOW" >> "$OUT"

# helper to run all .mac files in a dir with a physics array
run_macros_in_dir() {
  local dir="$1"; shift
  local -n phys_arr="$1"; shift
  if [ ! -d "$dir" ]; then
    return 0
  fi
  # iterate files robustly (sorted)
  while IFS= read -r -d '' mac; do
    echo -e "\n=== Running ${mac} (physics=${phys_arr[*]}) at $(date -u) ===" | tee -a "$OUT"
    "$APP" "${phys_arr[@]}" "$mac" >> "$OUT" 2>&1
    echo "=== Finished ${mac} at $(date -u) ===" >> "$OUT"
  done < <(find "$dir" -maxdepth 1 -type f -name '*.mac' -print0 | sort -z || true)
}

# run high then low
run_macros_in_dir "$MACRO_DIR_HIGH" PHYS_HIGH
run_macros_in_dir "$MACRO_DIR_LOW"  PHYS_LOW

echo -e "\nRun finished: $(date -u)" >> "$OUT"