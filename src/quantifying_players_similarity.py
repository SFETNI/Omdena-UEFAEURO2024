import os
import consts
import numpy as np
import pandas as pd


# Thomas Müller
PLAYER_ID = 5562
# Raquel Rodriguez
#PLAYER_ID = 4967
# Paulo Dybala
#PLAYER_ID = 5743
# Iago Aspas
#PLAYER_ID = 5217
# Marcos Paulo Mesquita Lopes(Rony Lopes)
PLAYER_ID2 = 3218
# Messi
#PLAYER_ID2 = 5503


def get_action_vector(shot_heatmap: np.array, pass_heatmap: np.array, dribble_heatmap: np.array,
                      player_column_index: int) -> np.array:
    shot_vector = shot_heatmap[:, player_column_index]
    pass_vector = pass_heatmap[:, player_column_index]
    dribble_vector = dribble_heatmap[:, player_column_index]

    return np.concat([shot_vector, pass_vector, dribble_vector])


def get_manhattan_distance(a, b):
    return np.abs(a - b).sum()


shot_action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_Shot_H.npy"))
pass_action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_Pass_H.npy"))
dribble_action_heatmap_H = np.load(os.path.join(consts.OUTPUT_DIR_PATH, f"compressed_heatmap_Dribble_H.npy"))
actions_grid = pd.read_csv(os.path.join(consts.OUTPUT_DIR_PATH, "actions_grid.csv"))
row_index = np.where(actions_grid["player_id"].unique() == PLAYER_ID)[0]
player_vector = get_action_vector(shot_action_heatmap_H, pass_action_heatmap_H, dribble_action_heatmap_H, row_index)
row_index2 = np.where(actions_grid["player_id"].unique() == PLAYER_ID2)[0]
player2_vector = get_action_vector(shot_action_heatmap_H, pass_action_heatmap_H, dribble_action_heatmap_H, row_index2)
manhattan_distance = get_manhattan_distance(player_vector, player2_vector)
print(f"Manhattan distance: {manhattan_distance}")

