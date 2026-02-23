#!/usr/bin/env python3
"""Generate framework diagram for MICCAI paper (no TikZ dependency)."""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.set_aspect('equal')
ax.axis('off')

def add_box(ax, xy, w, h, text, fill='white', edge='black'):
    r = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02", fill=fill, edgecolor=edge, linewidth=1.2)
    ax.add_patch(r)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center', fontsize=9)
    return r

def arrow(ax, start, end, color='black'):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Input
add_box(ax, (0.3, 1.0), 1.4, 0.7, '3D MRI\nBraTS/IXI', fill='#f0f0f0')
ax.text(1.0, 0.7, 'T2/FLAIR/T1c or T1+T2', ha='center', fontsize=7, style='italic')

# Frozen backbone
add_box(ax, (2.2, 0.9), 2.0, 1.0, 'Frozen Backbone\n$f_\\theta$ (UNet)', fill='#d0d0d0', edge='#666')
ax.text(3.2, 0.6, 'pretrained, fixed', ha='center', fontsize=7, style='italic')

# LoRA T2
add_box(ax, (4.8, 2.0), 1.3, 0.5, 'LoRA $\\phi_2$', fill='#cce5ff')
add_box(ax, (6.5, 2.0), 1.1, 0.5, 'Head $h_2$', fill='#cce5ff')
add_box(ax, (8.0, 2.0), 1.0, 0.5, 'Seg mask', fill='#e8f4fc')

# LoRA T3
add_box(ax, (4.8, 0.4), 1.3, 0.5, 'LoRA $\\phi_3$', fill='#cce5ff')
add_box(ax, (6.5, 0.4), 1.1, 0.5, 'Head $h_3$', fill='#cce5ff')
add_box(ax, (8.0, 0.4), 1.0, 0.5, 'Age (yr)', fill='#e8f4fc')

# Arrows
arrow(ax, (1.7, 1.35), (2.2, 1.4))
# backbone to lora2
ax.plot([4.2, 4.5, 4.5], [1.4, 1.4, 2.25], 'k-', lw=1.2)
ax.plot(4.5, 2.25, 'k>', markersize=8)
# backbone to lora3
ax.plot([4.2, 4.5, 4.5], [1.4, 1.4, 0.65], 'k-', lw=1.2)
ax.plot(4.5, 0.65, 'k>', markersize=8)
arrow(ax, (6.1, 2.25), (6.5, 2.25))
arrow(ax, (7.8, 2.25), (8.0, 2.25))
arrow(ax, (6.1, 0.65), (6.5, 0.65))
arrow(ax, (7.6, 0.65), (8.0, 0.65))

# Legend
ax.add_patch(mpatches.Rectangle((0.3, -0.2), 0.25, 0.15, fill=True, facecolor='#d0d0d0', edgecolor='#666'))
ax.text(0.7, -0.125, 'frozen', fontsize=8)
ax.add_patch(mpatches.Rectangle((2.0, -0.2), 0.25, 0.15, fill=True, facecolor='#cce5ff', edgecolor='black'))
ax.text(2.4, -0.125, 'trainable', fontsize=8)

plt.tight_layout()
base = os.path.dirname(os.path.dirname(__file__))
out_path = os.path.join(base, 'MICCAI_paper', 'figures', 'framework_diagram.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
print(f"Saved {out_path}")
plt.close()
