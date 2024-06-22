import os
import consts
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch, VerticalPitch, FontManager
from scipy.ndimage import gaussian_filter


FOOTBALL_PITCH_TILES = consts.ACTIONS_GRID_SHAPE
PLAYER_ID = 5574
ACTION_TYPE = "Pass"

action_heatmap = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"actions_heatmap_{ACTION_TYPE}.npy"))
action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{ACTION_TYPE}_H.npy"))
action_heatmap_W = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{ACTION_TYPE}_W.npy"))
reconstructed_action_heatmap = np.matmul(action_heatmap_W, action_heatmap_H)
actions_grid_smoothed = pd.read_csv(consts.START_ACTIONS_GRID_SMOOTHED_FILE_PATH)
row_index = np.where(actions_grid_smoothed["player_id"].unique() == PLAYER_ID)[0]
#actions = action_heatmap_W[:, 0]

# setup pitch
pitch = Pitch(pitch_type='statsbomb', line_zorder=2,
              pitch_color='#22312b', line_color='#efefef')
# draw
fig, ax = pitch.draw(figsize=(12, 8))
fig.set_facecolor('#22312b')

action_heatmap_flattened = action_heatmap[:, row_index].flatten()
start_plus_end_tiles = action_heatmap_flattened[:96]+action_heatmap_flattened[96:]
player_action_heatmap = start_plus_end_tiles
player_action_heatmap = player_action_heatmap.reshape(FOOTBALL_PITCH_TILES)

y, x = player_action_heatmap.shape
x_grid = np.linspace(0, 120, x + 1)
y_grid = np.linspace(0, 80, y + 1)
cx = x_grid[:-1] + 0.5 * (x_grid[1] - x_grid[0])
cy = y_grid[:-1] + 0.5 * (y_grid[1] - y_grid[0])
player_action_heatmap = gaussian_filter(player_action_heatmap, 1)
stats = dict(statistic=player_action_heatmap, x_grid=x_grid, y_grid=y_grid, cx=cx, cy=cy)
pcm = pitch.heatmap(stats, ax=ax, cmap='hot', edgecolors='#22312b')

# Add the colorbar and format off-white
cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
cbar.outline.set_edgecolor('#efefef')
cbar.ax.yaxis.set_tick_params(color='#efefef')
ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')

plt.show()
