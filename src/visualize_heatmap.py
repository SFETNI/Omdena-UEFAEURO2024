import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch, VerticalPitch, FontManager

FOOTBALL_PITCH_TILES = (12, 8)
FOOTBALL_PITCH_SIZE = (120, 80)
HEATMAP_TILE_SIZE = (10, 10)
shots_heatmap_H = np.load(r"C:\projects\ml\intro_to_ml_for_sport_analysis\output\compressed_heatmap_Dribble_H.npy")
shots_heatmap_W = np.load(r"C:\projects\ml\intro_to_ml_for_sport_analysis\output\compressed_heatmap_Dribble_W.npy")
shots_heatmap = np.matmul(shots_heatmap_H, shots_heatmap_W)

# setup pitch
pitch = Pitch(pitch_type='statsbomb', line_zorder=2,
              pitch_color='#22312b', line_color='#efefef')
# draw
fig, ax = pitch.draw(figsize=(6.6, 4.125))
fig.set_facecolor('#22312b')

dx = [5+i*10 for i in range(FOOTBALL_PITCH_SIZE[0])]
dy = [5+i*10 for i in range(FOOTBALL_PITCH_SIZE[0])]
values = shots_heatmap[0].reshape(FOOTBALL_PITCH_TILES)

bin_statistic = pitch.bin_statistic(dx, dy, values, statistic="sum", bins=FOOTBALL_PITCH_TILES)
pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
# Add the colorbar and format off-white
cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
cbar.outline.set_edgecolor('#efefef')
cbar.ax.yaxis.set_tick_params(color='#efefef')
ticks = plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#efefef')
