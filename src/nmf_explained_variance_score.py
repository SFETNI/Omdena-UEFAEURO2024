import os
import consts
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score


for action_type in ["Pass", "Dribble", "Shot", "Cross"]:
    action_heatmap = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"actions_heatmap_{action_type}.npy"))
    action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{action_type}_H.npy"))
    action_heatmap_W = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_{action_type}_W.npy"))
    reconstructed_action_heatmap = np.matmul(action_heatmap_W, action_heatmap_H)
    actions_grid_smoothed = pd.read_csv(os.path.join(consts.OUTPUT_DIR_PATH, "start_actions_grid_smoothed.csv"))
    score = explained_variance_score(action_heatmap, reconstructed_action_heatmap)
    mse = mean_squared_error(action_heatmap, reconstructed_action_heatmap)

    print(f"Explained variance score for action type '{action_type} is {score}.")
    print(f"MSE for action type '{action_type} is {mse}.")

