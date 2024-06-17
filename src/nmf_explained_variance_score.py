import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

# Thomas Müller
PLAYER_ID = 5562
# Raquel Rodriguez
#PLAYER_ID = 4967
# Paulo Dybala
#PLAYER_ID = 5743
# Messi
#PLAYER_ID = 5503
OUTPUT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..", "output")
ACTION_TYPE = "Dribble"
action_heatmap = np.load(os.path.join(OUTPUT_DIR_PATH, f"actions_heatmap_{ACTION_TYPE}.npy"))
action_heatmap_H = np.load(os.path.join(OUTPUT_DIR_PATH, f"compressed_heatmap_{ACTION_TYPE}_H.npy"))
action_heatmap_W = np.load(os.path.join(OUTPUT_DIR_PATH, f"compressed_heatmap_{ACTION_TYPE}_W.npy"))
reconstructed_action_heatmap = np.matmul(action_heatmap_H, action_heatmap_W)
actions_grid_smoothed = pd.read_csv(os.path.join(OUTPUT_DIR_PATH, "actions_grid_smoothed.csv"))
row_index = np.where(actions_grid_smoothed["player_id"].unique() == PLAYER_ID)[0]
#score = explained_variance_score(action_heatmap[row_index], reconstructed_action_heatmap[row_index])
#mse = mean_squared_error(action_heatmap[row_index], reconstructed_action_heatmap[row_index])
score = explained_variance_score(action_heatmap, reconstructed_action_heatmap)
mse = mean_squared_error(action_heatmap, reconstructed_action_heatmap)

print(f"Explained variance score for action type '{ACTION_TYPE} is {score}.")
print(f"MSE for action type '{ACTION_TYPE} is {mse}.")

