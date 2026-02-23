# MICCAI 2026 Paper: Few-Shot Continual Learning for 3D Brain MRI

**Target:** MICCAI 2026 | **Deadline:** Feb 26, 2026

## Structure (LaTeX, multi-file)

```
MICCAI_paper/
├── main.tex              # Main document (article class, local preview)
├── miccai_2026.tex       # MICCAI-style document (uses sections/, references.bib)
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── results.tex
│   ├── discussion.tex
│   └── conclusion.tex
├── references.bib
└── README.md
```

## Build

**main.tex (article, local preview):**
```bash
cd MICCAI_paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**miccai_2026.tex (paper in MICCAI-ready structure):**
```bash
cd MICCAI_paper
pdflatex miccai_2026 && bibtex miccai_2026 && pdflatex miccai_2026 && pdflatex miccai_2026
```

Or with latexmk: `latexmk -pdf main.tex` or `latexmk -pdf miccai_2026.tex`

## Template Note

`miccai_2026.tex` uses the article class for local compilation (lncs.cls not included). **Before submission**, download the official **MICCAI 2026 LaTeX template** from:

https://conferences.miccai.org/2026/files/downloads/MICCAI2026-Latex-Template.zip

Copy the preamble from the template's main.tex (or replace \documentclass with \documentclass[runningheads]{llncs}), ensure llncs.cls and splncs04.bst are present, and ensure:
- 8 pages max (text, figures, tables) + up to 2 pages references
- Anonymized author block (double-blind)
- No margin/spacing modifications

## Content Summary

- **Method:** Frozen FOMO backbone + task-specific LoRA adapters for continual learning
- **Results:** BWT=0 (LoRA) vs BWT≈-0.65 (Sequential FT); T2 Dice 0.62, T3 MAE 0.16 (32-shot)
- **Data:** BraTS 2023 Glioma, IXI; tasks T2 (segmentation), T3 (brain age)
