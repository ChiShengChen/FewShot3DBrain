#!/usr/bin/env bash
# Wait until the GPU is free, then run a command. Never pre-empts an existing job.
#
#   bash scripts/gpu_queue.sh [--mem-free MB] [--idle-checks N] [--poll SEC] -- <command...>
#
# The GPU counts as free when, for --idle-checks consecutive polls, no compute process
# other than this script's own children is resident AND at least --mem-free MB is
# available. Graphics processes (Xorg, the desktop, browsers) are ignored.

set -uo pipefail

MEM_FREE=12000      # MB of free VRAM required
IDLE_CHECKS=3       # consecutive clean polls before starting
POLL=120            # seconds between polls
MAX_WAIT=$((72 * 3600))
# Always-on services that hold VRAM but never finish. They are not jobs we can wait out,
# so they are excluded from the "is anyone else training?" check while still counting
# against free memory.
IGNORE_PATTERN='local-whisper|whisper|ollama|jupyter'
MAX_UTIL=25         # %, sustained utilisation below this counts as idle

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem-free)    MEM_FREE="$2"; shift 2 ;;
    --idle-checks) IDLE_CHECKS="$2"; shift 2 ;;
    --poll)        POLL="$2"; shift 2 ;;
    --max-wait)    MAX_WAIT="$2"; shift 2 ;;
    --ignore)      IGNORE_PATTERN="$2"; shift 2 ;;
    --max-util)    MAX_UTIL="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [opts] -- <command...>" >&2
  exit 2
fi

busy_pids() {
  # Compute-mode PIDs only; the desktop stack shows up as type G and does not block us.
  # Long-lived services matching IGNORE_PATTERN are filtered out by cmdline.
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)
  for p in $pids; do
    local args
    args=$(ps -o args= -p "$p" 2>/dev/null || true)
    [[ -z "$args" ]] && continue
    if [[ -n "$IGNORE_PATTERN" ]] && echo "$args" | grep -qE "$IGNORE_PATTERN"; then
      continue
    fi
    echo "$p"
  done
}

gpu_util() {
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1
}

free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

describe_busy() {
  for p in $(busy_pids); do
    printf '    pid %s: %s\n' "$p" "$(ps -o args= -p "$p" 2>/dev/null | cut -c1-100)"
  done
}

echo "[gpu-queue] $(date '+%F %T') waiting for GPU (need ${MEM_FREE}MB free, <=${MAX_UTIL}% util, ${IDLE_CHECKS} clean polls, every ${POLL}s)"
echo "[gpu-queue] ignoring persistent services matching: ${IGNORE_PATTERN:-<none>}"

clean=0
waited=0
while (( waited < MAX_WAIT )); do
  pids=$(busy_pids)
  mem=$(free_mb)
  util=$(gpu_util)
  if [[ -z "$pids" && -n "$mem" && "$mem" -ge "$MEM_FREE" && -n "$util" && "$util" -le "$MAX_UTIL" ]]; then
    clean=$((clean + 1))
    echo "[gpu-queue] $(date '+%F %T') GPU idle (${mem}MB free, ${util}% util) ${clean}/${IDLE_CHECKS}"
    if (( clean >= IDLE_CHECKS )); then
      break
    fi
  else
    if (( clean > 0 )); then
      echo "[gpu-queue] $(date '+%F %T') GPU busy again, resetting counter"
    fi
    clean=0
    echo "[gpu-queue] $(date '+%F %T') busy — ${mem:-?}MB free, ${util:-?}% util, blocking pids: ${pids:-none}"
    describe_busy
  fi
  sleep "$POLL"
  waited=$((waited + POLL))
done

if (( clean < IDLE_CHECKS )); then
  echo "[gpu-queue] gave up after ${MAX_WAIT}s of waiting" >&2
  exit 1
fi

echo "[gpu-queue] $(date '+%F %T') starting: $*"
exec "$@"
