import os
import math
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


ACTIONS_GRID_SHAPE = (60, 40)
OUTPUT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..", "output")
ACTIONS_GRID_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "actions_grid.csv")
ACTIONS_GRID_NORMALIZED_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "actions_grid_normalized.csv")
ACTIONS_GRID_SMOOTHED_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "actions_grid_smoothed.csv")
PLAYERS_TOTAL_PLAYED_TIME_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "players_total_played_time.csv")
COMPONENTS_PER_ACTION = { "Shot": 10, "Pass": 10, "Dribble": 10 }


def load_data(actions_grid_file_path: str, players_total_played_time_file_path: str, action_grid_shape: tuple[int, int]):
    df_actions_grid = pd.read_csv(actions_grid_file_path, converters={
        "player_id": int, "Pass": float, "Show": float, "Dribble": float, "grid_index": int})
    df_players_total_played_time = pd.read_csv(players_total_played_time_file_path)
    #df_actions_grid = df_actions_grid.astype({"Pass": float, "Shot": float, "Dribble": float})
    # Remove players from the actions grid that are not found in the players total played time dataframe
    # See extract_players_played_time.py for the details why is this necessary.
    unique_player_ids = df_players_total_played_time["player_id"].unique()
    df_actions_grid = df_actions_grid.loc[df_actions_grid["player_id"].isin(unique_player_ids)]
    max_tiles = action_grid_shape[0] * action_grid_shape[1]
    df_players = pd.DataFrame(unique_player_ids, columns=["player_id"])
    df_grids = pd.DataFrame(range(max_tiles), columns=["grid_index"])
    df_players_grids = df_players.join(df_grids, how="cross")

    df_actions_grid = df_actions_grid.merge(df_players_grids, how="right", on=["player_id", "grid_index"])
    df_actions_grid[["Shot", "Pass", "Dribble"]] = df_actions_grid[["Shot", "Pass", "Dribble"]].fillna(0)

    return df_actions_grid, df_players_total_played_time


def normalize_actions_data(df_actions_grid: pd.DataFrame, df_players_total_played_time: pd.DataFrame):
    df_actions_grid_normalized = df_actions_grid.copy()
    df_actions_grid_normalized[["Shot", "Pass", "Dribble"]] = df_actions_grid_normalized.apply(
        normalize_actions_data_row, axis=1, args=(df_players_total_played_time,))

    return df_actions_grid_normalized


def normalize_actions_data_row(row, df_players_total_played_time: pd.DataFrame):
    total_played_time = df_players_total_played_time.loc[
        df_players_total_played_time["player_id"] == row["player_id"]]["play_duration"].values[0]

    return [(1/total_played_time)*row["Shot"],
            (1/total_played_time)*row["Pass"],
            (1/total_played_time)*row["Dribble"]]


def get_gaussian_kernel(kernel_radius=1):
    # Function implementation found at https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
    gaussian = lambda x, mu, sigma: math.exp(-(((x - mu) / (sigma)) ** 2) / 2.0)

    sigma = kernel_radius / 2.  # for [-2*sigma, 2*sigma]

    # compute the actual kernel elements
    h_kernel = [gaussian(x, kernel_radius, sigma) for x in range(2 * kernel_radius + 1)]
    v_kernel = [x for x in h_kernel]
    kernel2d = [[xh * xv for xh in h_kernel] for xv in v_kernel]

    # normalize the kernel elements
    kernel_sum = sum([sum(row) for row in kernel2d])
    kernel2d = np.array([[x / kernel_sum for x in row] for row in kernel2d])

    return kernel2d


def smooth_action(df_player_action_grid: pd.DataFrame, action_type: str, actions_grid_matrix_shape: tuple, smoothing_kernel: np.array):
    kernel_half_width = int(smoothing_kernel.shape[0]/2)
    kernel_half_height = int(smoothing_kernel.shape[1]/2)
    player_id = df_player_action_grid["player_id"].unique()[0]

    action_type_matrix = df_player_action_grid[action_type].values.reshape(
        actions_grid_matrix_shape[0], actions_grid_matrix_shape[1])
    temp_matrix = np.zeros([actions_grid_matrix_shape[0] + kernel_half_width*2,
                           actions_grid_matrix_shape[1] + kernel_half_height*2])
    temp_matrix[kernel_half_width:temp_matrix.shape[0] - kernel_half_width,
                kernel_half_height:temp_matrix.shape[1] - kernel_half_height] = action_type_matrix
    smoothed_shots_matrix = np.zeros(action_type_matrix.shape)
    for i in range(kernel_half_width, temp_matrix.shape[0] - kernel_half_width):
        for j in range(kernel_half_height, temp_matrix.shape[1] - kernel_half_height):
            smoothed_shots_matrix[i - kernel_half_width, j - kernel_half_height] = (
                (temp_matrix[i - kernel_half_width:i + kernel_half_width + 1,
                 j - kernel_half_height:j + kernel_half_height + 1] * smoothing_kernel).sum())
    smoothed_data = np.reshape(smoothed_shots_matrix, actions_grid_matrix_shape[0] * actions_grid_matrix_shape[1])

    return smoothed_data

def smooth_actions_data(df_actions_grid: pd.DataFrame, actions_grid_matrix_shape: tuple):
    kernel_matrix = get_gaussian_kernel()
    player_ids = df_actions_grid["player_id"].unique()
    df_actions_grid_smoothed = df_actions_grid.drop(df_actions_grid.index)

    for player_id in player_ids:
        df_player_action_grid = df_actions_grid.loc[df_actions_grid["player_id"] == player_id].copy()
        for action_type in ["Shot", "Pass", "Dribble"]:
            smoothed_data = smooth_action(df_player_action_grid, action_type, actions_grid_matrix_shape, kernel_matrix)
            df_player_action_grid[action_type] = smoothed_data
        df_actions_grid_smoothed = pd.concat([df_actions_grid_smoothed, df_player_action_grid], ignore_index=True)

    return df_actions_grid_smoothed


def compress_heatmap(df_actions_grid: pd.DataFrame, components_per_action: dict, max_iter: int=1000):
    actions_heatmap = {}
    action_types =  ["Dribble", "Shot", "Pass"]

    for player_id in df_actions_grid["player_id"].unique():

        for action_type in action_types:
            player_actions = df_actions_grid.loc[df_actions_grid["player_id"] == player_id][action_type].to_numpy()
            if action_type not in actions_heatmap:
                #actions_heatmap[action_type] = player_actions
                actions_heatmap[action_type] = player_actions.reshape((player_actions.shape[0], 1))
            else:
                #actions_heatmap[action_type] = np.vstack([actions_heatmap[action_type], player_actions])
                actions_heatmap[action_type] = np.hstack([actions_heatmap[action_type], player_actions.reshape((player_actions.shape[0], 1))])

    compressed_actions_heatmap = {}
    for action_type, heatmap in actions_heatmap.items():
        model = NMF(n_components=components_per_action[action_type], init='random', random_state=0, max_iter=max_iter)
        W = model.fit_transform(heatmap)
        H = model.components_
        compressed_actions_heatmap[action_type] = (W, H)
        print(f"NMF reconstruction error for action '{action_type}: {model.reconstruction_err_}")

    return actions_heatmap, compressed_actions_heatmap


def save_heatmaps(action_heatmap: dict[str, np.array],
                  compressed_actions_heatmap: dict[str, np.array],
                  output_dir_path: str) -> None:
    for action_type, (W, H) in compressed_actions_heatmap.items():
        np.save(os.path.join(output_dir_path, f"compressed_heatmap_{action_type}_W"), W)
        np.save(os.path.join(output_dir_path, f"compressed_heatmap_{action_type}_H"), H)
    for action_type, heatmap in action_heatmap.items():
        np.save(os.path.join(output_dir_path, f"actions_heatmap_{action_type}"), heatmap)


if __name__ == "__main__":
    start_time = time.time()
    df_actions_grid, df_players_total_played_time = load_data(ACTIONS_GRID_FILE_PATH,
                                                              PLAYERS_TOTAL_PLAYED_TIME_FILE_PATH,
                                                              ACTIONS_GRID_SHAPE)
    #df_actions_grid = df_actions_grid.loc[df_actions_grid["player_id"] == 3567]
    #$df_players_total_played_time = df_players_total_played_time.loc[df_players_total_played_time["player_id"] == 3567]
    """
    df_actions_grid_normalized = df_actions_grid.merge(df_players_total_played_time, on="player_id", how="inner", validate="m:1")
    df_actions_grid_normalized["Shot"] /= df_actions_grid_normalized["play_duration"]
    df_actions_grid_normalized["Pass"] /= df_actions_grid_normalized["play_duration"]
    df_actions_grid_normalized["Dribble"] /= df_actions_grid_normalized["play_duration"]
    #df_actions_grid_normalized = normalize_actions_data(df_actions_grid_temp, df_players_total_played_time)
    df_actions_grid_normalized.drop(columns=["play_duration"], inplace=True)
    df_actions_grid_normalized.to_csv(ACTIONS_GRID_NORMALIZED_FILE_PATH, index=False)
    df_actions_grid_smoothed = smooth_actions_data(df_actions_grid_normalized, actions_grid_matrix_shape=ACTIONS_GRID_SHAPE)
    df_actions_grid_smoothed.to_csv(ACTIONS_GRID_SMOOTHED_FILE_PATH, index=False)
    """
    actions_heatmap, compressed_actions_heatmap = compress_heatmap(df_actions_grid, COMPONENTS_PER_ACTION)
    save_heatmaps(actions_heatmap, compressed_actions_heatmap, OUTPUT_DIR_PATH)
    end_time = time.time()
    print(f"Load events elapsed time: {end_time - start_time}s")
