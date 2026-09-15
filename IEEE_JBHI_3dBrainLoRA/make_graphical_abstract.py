"""
Generate Graphical Abstract for IEEE JBHI submission.
Layout:
  Row 0: Title
  Row 1: [Seg Input | Seg Overlay] [Framework Diagram] [Age Axial | Age Sagittal]
  Row 2: Three key-finding cards
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

# ── Load source images ──────────────────────────────────────────────
framework = Image.open("figures/framework_diagram.png")
seg_full  = Image.open("figures/show_case_task2.png")
age_full  = Image.open("figures/task3_age_s42_stack6.png")

# ── Crop helper ─────────────────────────────────────────────────────
def crop_grid_cell(img, row, col, n_rows, n_cols, top_margin=0.08, left_margin=0.0):
    w, h = img.size
    cell_w = w / n_cols
    cell_h = (h * (1 - top_margin)) / n_rows
    y_offset = int(h * top_margin)
    x0 = int(col * cell_w + left_margin * w)
    y0 = int(y_offset + row * cell_h)
    x1 = int((col + 1) * cell_w)
    y1 = int(y_offset + (row + 1) * cell_h)
    return img.crop((x0, y0, x1, y1))

# Crop segmentation examples (row 2 = nice tumor case)
seg_input   = crop_grid_cell(seg_full, row=2, col=0, n_rows=6, n_cols=4, top_margin=0.06)
seg_overlay = crop_grid_cell(seg_full, row=2, col=3, n_rows=6, n_cols=4, top_margin=0.06)

# Crop age regression examples
age_axial = crop_grid_cell(age_full, row=2, col=0, n_rows=6, n_cols=4, top_margin=0.05, left_margin=0.02)
age_sagit = crop_grid_cell(age_full, row=2, col=2, n_rows=6, n_cols=4, top_margin=0.05)

# ── Figure setup ────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor="white")

gs = fig.add_gridspec(
    4, 6,
    height_ratios=[0.07, 0.03, 0.55, 0.35],
    width_ratios=[1, 1, 1.6, 1.6, 1, 1],
    hspace=0.08, wspace=0.10,
    left=0.04, right=0.96, top=0.96, bottom=0.04
)

# ── Row 0: Title ────────────────────────────────────────────────────
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis("off")
ax_title.text(
    0.5, 0.5,
    "Few-Shot Continual Learning for 3D Brain MRI with Frozen Foundation Models",
    ha="center", va="center",
    fontsize=20, fontweight="bold", color="#1a1a2e",
)

# ── Row 1: Task labels ─────────────────────────────────────────────
ax_lab_l = fig.add_subplot(gs[1, 0:2])
ax_lab_l.axis("off")
ax_lab_l.text(0.5, 0.5, "Task 1: Tumor Segmentation",
              ha="center", va="center", fontsize=13, fontweight="bold", color="#c0392b",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fadbd8", edgecolor="#c0392b", alpha=0.9),
              transform=ax_lab_l.transAxes)

ax_lab_c = fig.add_subplot(gs[1, 2:4])
ax_lab_c.axis("off")
ax_lab_c.text(0.5, 0.5, "Proposed Framework",
              ha="center", va="center", fontsize=13, fontweight="bold", color="#2c3e50",
              transform=ax_lab_c.transAxes)

ax_lab_r = fig.add_subplot(gs[1, 4:6])
ax_lab_r.axis("off")
ax_lab_r.text(0.5, 0.5, "Task 2: Brain Age Estimation",
              ha="center", va="center", fontsize=13, fontweight="bold", color="#2471a3",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4e6f1", edgecolor="#2471a3", alpha=0.9),
              transform=ax_lab_r.transAxes)

# ── Row 2: Main content ────────────────────────────────────────────
# Left: seg input
ax_seg_in = fig.add_subplot(gs[2, 0])
ax_seg_in.imshow(seg_input)
ax_seg_in.set_title("Input (T2-FLAIR)", fontsize=10, fontweight="bold", color="#555")
ax_seg_in.axis("off")
# Add red border
for spine in ax_seg_in.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#c0392b")
    spine.set_linewidth(2)

# Left: seg overlay
ax_seg_ov = fig.add_subplot(gs[2, 1])
ax_seg_ov.imshow(seg_overlay)
ax_seg_ov.set_title("Prediction Overlay", fontsize=10, fontweight="bold", color="#555")
ax_seg_ov.axis("off")
for spine in ax_seg_ov.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#c0392b")
    spine.set_linewidth(2)

# Center: framework
ax_fw = fig.add_subplot(gs[2, 2:4])
ax_fw.imshow(framework)
ax_fw.axis("off")

# Right: age axial
ax_age_ax = fig.add_subplot(gs[2, 4])
ax_age_ax.imshow(age_axial)
ax_age_ax.set_title("Axial View", fontsize=10, fontweight="bold", color="#555")
ax_age_ax.axis("off")
for spine in ax_age_ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#2471a3")
    spine.set_linewidth(2)

# Right: age sagittal
ax_age_sg = fig.add_subplot(gs[2, 5])
ax_age_sg.imshow(age_sagit)
ax_age_sg.set_title("Sagittal View", fontsize=10, fontweight="bold", color="#555")
ax_age_sg.axis("off")
for spine in ax_age_sg.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("#2471a3")
    spine.set_linewidth(2)

# Add age prediction annotation on the sagittal view
ax_age_sg.text(0.5, -0.08, "Pred: 46.0 yr | GT: 50.0 yr",
               ha="center", va="top", fontsize=9, color="#2471a3", fontweight="bold",
               transform=ax_age_sg.transAxes)

# ── Row 3: Key findings banner ──────────────────────────────────────
findings = [
    {
        "icon": "BWT = 0",
        "title": "Zero Catastrophic Forgetting",
        "desc": "Frozen backbone + isolated\nLoRA adapters guarantee\nno forgetting by design",
        "color": "#27ae60",
        "bg": "#eafaf1",
    },
    {
        "icon": "< 0.1%",
        "title": "Parameter Efficient",
        "desc": "Only ~28K-47K trainable params\nper task (vs. 59M backbone)\n~1.7 GB peak GPU memory",
        "color": "#e67e22",
        "bg": "#fef5e7",
    },
    {
        "icon": "n = 32",
        "title": "Few-Shot Ready",
        "desc": "Competitive performance with\nas few as 32 labeled samples\nper downstream task",
        "color": "#8e44ad",
        "bg": "#f5eef8",
    },
]

for i, f in enumerate(findings):
    ax = fig.add_subplot(gs[3, i*2:(i+1)*2])
    ax.axis("off")

    # Card background
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.03, 0.05), 0.94, 0.90,
        boxstyle="round,pad=0.03",
        facecolor=f["bg"], edgecolor=f["color"], linewidth=2.5,
        transform=ax.transAxes
    ))

    # Icon / number
    ax.text(0.5, 0.82, f["icon"],
            ha="center", va="center",
            fontsize=22, fontweight="bold", color=f["color"],
            transform=ax.transAxes,
            fontfamily="monospace")

    # Title
    ax.text(0.5, 0.58, f["title"],
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="#2c3e50",
            transform=ax.transAxes)

    # Description
    ax.text(0.5, 0.28, f["desc"],
            ha="center", va="center",
            fontsize=10, color="#555",
            transform=ax.transAxes,
            linespacing=1.3)

# ── Arrows: framework -> tasks ──────────────────────────────────────
# Left arrow
fig.patches.append(mpatches.FancyArrowPatch(
    (0.35, 0.52), (0.29, 0.52),
    transform=fig.transFigure,
    arrowstyle="->,head_width=10,head_length=8",
    color="#c0392b", linewidth=3,
    mutation_scale=1
))
# Right arrow
fig.patches.append(mpatches.FancyArrowPatch(
    (0.65, 0.52), (0.71, 0.52),
    transform=fig.transFigure,
    arrowstyle="->,head_width=10,head_length=8",
    color="#2471a3", linewidth=3,
    mutation_scale=1
))

# ── Save ────────────────────────────────────────────────────────────
output_path = "graphical_abstract.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("graphical_abstract.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("graphical_abstract_preview.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {output_path} ({Image.open(output_path).size})")
print("Saved: graphical_abstract.pdf")
print("Saved: graphical_abstract_preview.png")
plt.close()

# ── Graphical Abstract Text PDF ─────────────────────────────────────
fig2 = plt.figure(figsize=(8, 3), facecolor="white")
ax2 = fig2.add_subplot(111)
ax2.axis("off")

abstract_text = (
    "We propose a parameter-efficient continual learning framework for 3D brain MRI "
    "that freezes a pretrained foundation model and attaches task-specific LoRA adapters. "
    "This architecture eliminates catastrophic forgetting by design (BWT=0) while enabling "
    "few-shot adaptation to heterogeneous tasks including tumor segmentation and brain age estimation."
)

ax2.text(
    0.5, 0.5, abstract_text,
    ha="center", va="center",
    fontsize=14, color="#1a1a2e",
    wrap=True,
    transform=ax2.transAxes,
    linespacing=1.6,
    fontfamily="serif",
    bbox=dict(boxstyle="square,pad=0.5", facecolor="white", edgecolor="white")
)

fig2.savefig("graphical_abstract_text.pdf", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved: graphical_abstract_text.pdf")
plt.close()
