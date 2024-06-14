import os
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch, VerticalPitch, FontManager
from scipy.ndimage import gaussian_filter

FOOTBALL_PITCH_TILES = (30, 20)
FOOTBALL_PITCH_SIZE = (120, 80)
HEATMAP_TILE_SIZE = (4, 4)
PLAYER_ID = 10650
OUTPUT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..", "output")
actions_heatmap = np.load(os.path.join(OUTPUT_DIR_PATH, "actions_heatmap_Shot.npy"))
shots_heatmap_H = np.load(os.path.join(OUTPUT_DIR_PATH, "compressed_heatmap_Shot_H.npy"))
shots_heatmap_W = np.load(os.path.join(OUTPUT_DIR_PATH, "compressed_heatmap_Shot_W.npy"))
reconstructed_shots_heatmap = np.matmul(shots_heatmap_H, shots_heatmap_W)
actions_grid_smoothed = pd.read_csv(os.path.join(OUTPUT_DIR_PATH, "actions_grid_smoothed.csv"))
row_index = np.where(actions_grid_smoothed["player_id"].unique() == PLAYER_ID)[0]
shots_heatmap = actions_grid_smoothed.loc[actions_grid_smoothed["player_id"] == PLAYER_ID]["Shot"].to_numpy()

# setup pitch
pitch = Pitch(pitch_type='statsbomb', line_zorder=2,
              pitch_color='#22312b', line_color='#efefef')
# draw
fig, ax = pitch.draw(figsize=(12, 8))
fig.set_facecolor('#22312b')

player_shots_heatmap = reconstructed_shots_heatmap[row_index]#shots_heatmap#actions_heatmap[row_index]
player_shots_heatmap = player_shots_heatmap.reshape(FOOTBALL_PITCH_TILES)

y, x = player_shots_heatmap.shape
x_grid = np.linspace(0, 120, x + 1)
y_grid = np.linspace(0, 80, y + 1)
cx = x_grid[:-1] + 0.5 * (x_grid[1] - x_grid[0])
cy = y_grid[:-1] + 0.5 * (y_grid[1] - y_grid[0])
smoothed_player_shots_heatmap = gaussian_filter(player_shots_heatmap, 1)
stats = dict(statistic=smoothed_player_shots_heatmap, x_grid=x_grid, y_grid=y_grid, cx=cx, cy=cy)
pcm = pitch.heatmap(stats, ax=ax, cmap='hot', edgecolors='#22312b')

# Add the colorbar and format off-white
cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
cbar.outline.set_edgecolor('#efefef')
cbar.ax.yaxis.set_tick_params(color='#efefef')
ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

plt.show()
