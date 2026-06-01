"""Generate schematic field images for the thesis."""


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


n = 128
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

a = (1.0
     + 3.0 * np.exp(-((X - 0.3)**2 + (Y - 0.2)**2) / 0.08)
     + 1.5 * np.exp(-((X - 0.75)**2 + (Y - 0.7)**2) / 0.05)
     - 0.8 * np.exp(-((X - 0.6)**2 + (Y - 0.3)**2) / 0.12))

fig, ax = plt.subplots(1, 1, figsize=(2, 2))
ax.pcolormesh(X, Y, a, cmap='jet', shading='auto')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(out_dir / 'schematic_a.png', dpi=200, bbox_inches='tight', pad_inches=0)
plt.close()

u = (0.4 * np.sin(np.pi * X) * np.sin(np.pi * Y)
     + 0.15 * np.sin(2 * np.pi * X) * np.sin(np.pi * Y)
     + 0.1 * np.sin(np.pi * X) * np.sin(2 * np.pi * Y))

fig, ax = plt.subplots(1, 1, figsize=(2, 2))
ax.pcolormesh(X, Y, u, cmap='inferno', shading='auto')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(out_dir / 'schematic_u.png', dpi=200, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"Saved schematic_a.png and schematic_u.png to {out_dir}")
