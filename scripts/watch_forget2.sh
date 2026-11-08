#!/usr/bin/env bash
# Wait until the five FORGET1 runs (sequential_ft_t25_ownlr) have results, then queue FORGET2 (EWC from them).
cd "$(dirname "$0")/.."
while :; do
  n=$(ls outputs/benchmark_v2/n32/sequential_ft_t25_ownlr/seed4{2,3,4,5,6}/results.json 2>/dev/null | wc -l)
  if [ "$n" -eq 5 ]; then bash scripts/submit_reframe.sh FORGET2; exit 0; fi
  sleep 600
done
