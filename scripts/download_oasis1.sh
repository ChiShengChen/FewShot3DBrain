#!/usr/bin/env bash
# Download OASIS-1 (cross-sectional, 416 subjects, 18-96 y, T1 with age / sex / CDR).
#
# The twelve disc tarballs (~1.4 GB each, ~16.5 GB total) are served from the OASIS
# download host. Use of the data is governed by the OASIS Data Use Agreement at
# https://sites.wustl.edu/oasisbrains/ -- the person running this must have accepted it.
# Each subject folder OAS1_XXXX_MR1/ contains RAW/ (3-4 MPRAGE repeats, Analyze .hdr/.img),
# PROCESSED/ (atlas-registered, masked) and OAS1_XXXX_MR1.txt with AGE, M/F, CDR, MMSE.
#
#   bash scripts/download_oasis1.sh            # -> data/plan_b_raw/oasis1/
set -uo pipefail
cd "$(dirname "$0")/.."
DEST="${DEST:-data/plan_b_raw/oasis1}"
mkdir -p "$DEST/tar"
BASE=https://download.nrg.wustl.edu/data
for i in $(seq 1 12); do
  f="oasis_cross-sectional_disc${i}.tar.gz"
  if [[ -f "$DEST/tar/$f.ok" ]]; then echo "skip $f"; continue; fi
  for attempt in 1 2 3 4 5; do
    curl -fL --retry 5 --retry-delay 10 -C - -o "$DEST/tar/$f" "$BASE/$f" && break
    echo "retry $attempt for $f"; sleep $((attempt * 15))
  done
  tar -tzf "$DEST/tar/$f" > /dev/null || { echo "corrupt $f"; exit 1; }
  tar -xzf "$DEST/tar/$f" -C "$DEST"
  touch "$DEST/tar/$f.ok"
  echo "[$i/12] extracted $f  subjects so far: $(ls -d "$DEST"/disc*/OAS1_* 2>/dev/null | wc -l)"
done
cat > "$DEST/SOURCE.txt" <<EOF
OASIS-1 cross-sectional, https://sites.wustl.edu/oasisbrains/ (Data Use Agreement applies).
Marcus et al., J Cogn Neurosci 19(9) 2007, doi:10.1162/jocn.2007.19.9.1498
Grants to acknowledge: P50 AG05681, P01 AG03991, R01 AG021910, P20 MH071616, U24 RR021382
EOF
echo "done: $(ls -d "$DEST"/disc*/OAS1_* 2>/dev/null | wc -l) subject folders"
