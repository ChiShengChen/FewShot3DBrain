#!/usr/bin/env bash
# Build the JBHI/Overleaf zip and the arXiv tarball from IEEE_JBHI_3dBrainLoRA/main.tex.
# Assumes main.pdf / main_arxiv.pdf (and their .bbl) are freshly built. Usage: bash scripts/pack_submission.sh
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)/IEEE_JBHI_3dBrainLoRA"; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cd "$R"
deps() { grep -o 'includegraphics\(\[[^]]*\]\)\?{[^}]*}\|input{[^}]*}' main.tex | sed 's/.*{\(.*\)}/\1/' | sort -u; }
COMMON="reference.bib IEEEtran.cls IEEEtran.bst ieeecolor.cls algorithm.sty algorithmic.sty generic.sty logo.eps"
for f in $COMMON main.bbl main_arxiv.bbl $(deps); do [ -f "$f" ] || { echo "MISSING $f" >&2; exit 1; }; done
mkdir -p "$W/ovl" "$W/arx"
cp --parents main.tex main.bbl $COMMON $(deps) "$W/ovl/"
(cd "$W/ovl" && zip -qr "$R/FewShot3DBrain_JBHI_overleaf.zip" .)
cp --parents $COMMON $(deps) "$W/arx/"; cp main.tex "$W/arx/body.tex"; cp main_arxiv.bbl "$W/arx/"
printf '%s\n' '% Full preprint build: `pdflatex main_arxiv` (then bibtex main_arxiv, pdflatex x2).' '\def\arxivfull{1}' '\input{body}' > "$W/arx/main_arxiv.tex"
(cd "$W/arx" && tar czf "$R/FewShot3DBrain_arxiv.tar.gz" .)
cp main.pdf FewShot3DBrain_JBHI_manuscript.pdf
echo "packed: FewShot3DBrain_JBHI_overleaf.zip FewShot3DBrain_arxiv.tar.gz FewShot3DBrain_JBHI_manuscript.pdf"
