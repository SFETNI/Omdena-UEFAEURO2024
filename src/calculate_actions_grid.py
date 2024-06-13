import os

os.environ["OPEN_DATA_REPO_PATH"] = r"C:\projects\statsbomb_open_data"

import time
import queue
import logging
import pandas as pd
import numpy as np
import multiprocessing as mp
from statsbombpy_local import sb
from concurrent.futures import ProcessPoolExecutor

pd.options.mode.copy_on_write = True
logger = logging.getLogger(__name__)
STATSBOMB_OPEN_DATA_LOCAL_PATH = "C:\projects\statsbomb_open_data"
FOOTBALL_PITCH_SIZE = (120, 80)
HEATMAP_TILE_SIZE = (10, 10)


def fill_actions_grid(row: object, actions_grid: pd.DataFrame, tile_width: int, tile_height: int, max_horizontal_tiles: int):
    # We need to vertically and horizontally limit the x and y position to avoid issue with out of range index
    # when x and y pos are exactly on the pitch edges.
    x_loc = min(119, row["location"][0])
    y_loc = min(79, row["location"][1])
    horizontal_tile_index = int(x_loc/tile_width)
    vertical_tile_index = int(y_loc/tile_height)
    grid_index = vertical_tile_index * max_horizontal_tiles + horizontal_tile_index
    if grid_index > 95:
        logger.debug("grid_index > 95")
        logger.debug(row)
    action_grid_filter = (actions_grid["player_id"] == row["player_id"]) & (actions_grid["grid_index"] == grid_index)
    if len(actions_grid.loc[action_grid_filter].index) == 0:
        print(f"row: {row}")
    actions_grid_index = actions_grid.loc[action_grid_filter].index[0]
    actions_grid.at[actions_grid_index, row["type"]] += 1


def load_events_worker(matches_queue: queue.Queue, events_queue: queue.Queue):
    df_all_events = pd.DataFrame(np.array([], dtype=[("player_id", np.int32),
                                                     ("type", str), ("pass_cross", bool), ("location", object)]),
                                 columns=["player_id", "type", "pass_cross", "location"])
    matches_ids = matches_queue.get()

    for match_id in matches_ids:
        print(f"Loading events data for match with id: {match_id}")
        df_events = sb.events(match_id=match_id)
        # Filter out starting events such as kick off event.
        df_events = df_events.loc[df_events["player_id"].notna()]
        df_events["player_id"] = df_events["player_id"].astype(int)
        if df_all_events is None:
            df_all_events = df_events[["player_id", "type", "pass_cross", "location"]]
        else:
            df_all_events = pd.concat([df_all_events, df_events[["player_id", "type", "pass_cross", "location"]]])

    events_queue.put(df_all_events)
    matches_queue.task_done()


def load_events_worker_single_proc(matches_ids):
    df_all_events = pd.DataFrame(np.array([], dtype=[("player_id", np.int32),
                                                     ("type", str), ("pass_cross", bool), ("location", object)]),
                                 columns=["player_id", "type", "pass_cross", "location"])

    for match_id in matches_ids:
        print(f"Loading events data for match with id: {match_id}")
        df_events = sb.events(match_id=match_id)
        # Filter out starting events such as kick off event.
        df_events = df_events.loc[df_events["player_id"].notna()]
        df_events["player_id"] = df_events["player_id"].astype(int)
        if df_all_events is None:
            df_all_events = df_events[["player_id", "type", "pass_cross", "location"]]
        else:
            df_all_events = pd.concat([df_all_events, df_events[["player_id", "type", "pass_cross", "location"]]])

    return df_all_events


def load_events(matches_ids: list[int], parallel_processes_count=os.cpu_count()):
    logger.debug(f"matches count: {len(matches_ids)}")
    matches_slice_size = round(len(matches_ids)/parallel_processes_count)
    logger.debug(f"matches_slice_size: {matches_slice_size}")

    with mp.Manager() as manager:
        events_queue = manager.Queue()
        matches_queue = manager.Queue()
        for i in range(parallel_processes_count):
            start = i * matches_slice_size
            end = (i + 1) * matches_slice_size
            logger.debug(f"Start slice: {start}, end slice: {end}")
            matches_queue.put(matches_ids[start:end])
        logger.debug("Starting with events loading ...")
        with ProcessPoolExecutor(max_workers=parallel_processes_count) as executor:
            executor.map(load_events_worker, [matches_queue]*parallel_processes_count, [events_queue]*parallel_processes_count)
        matches_queue.join()

        df_all_events = None
        while not events_queue.empty():
            df_events = events_queue.get()
            if df_all_events is None:
                df_all_events = df_events
            else:
                df_all_events = pd.concat([df_all_events, df_events])

    return df_all_events


def load_events_single_proc(matches_ids: list[int]):
    logger.debug(f"matches count: {len(matches_ids)}")
    matches_slice_size = round(len(matches_ids))
    logger.debug(f"matches_slice_size: {matches_slice_size}")

    df_all_events = load_events_worker_single_proc(matches_ids)

    return df_all_events


def get_matches_ids(data_path: str, max_events=-1):
    matches_ids = []

    for idx, data_file in enumerate(os.listdir(data_path)):
        match_id = int(data_file.split(".")[0])
        matches_ids.append(match_id)
        if max_events != -1 and idx+1 == max_events:
            break

    return matches_ids


def get_actions(df_events: pd.DataFrame):
    action_types_filter = ((df_events.type == "Dribble") | (df_events.type == "Shot") |
                           ((df_events.type == "Pass") & df_events.pass_cross))
    df_actions = df_events.loc[action_types_filter].reset_index()

    return df_actions


def create_actions_grid(df_actions: pd.DataFrame, football_pitch_size, heatmap_tile_size):
    actions_grid_width = int(football_pitch_size[0] / heatmap_tile_size[0])
    actions_grid_height = int(football_pitch_size[1] / heatmap_tile_size[1])
    df_actions_grid_positions = pd.DataFrame(np.array(range(actions_grid_width * actions_grid_height)), columns=["grid_index"])
    df_actions_grid = df_actions[["player_id"]].drop_duplicates()
    df_actions_grid[np.array(df_actions.type.unique())] = 0
    df_actions_grid = df_actions_grid.join(df_actions_grid_positions, how="cross")
    df_actions.apply(fill_actions_grid, axis=1, args=(df_actions_grid, heatmap_tile_size[0], heatmap_tile_size[1],
                                                      int(football_pitch_size[0] / heatmap_tile_size[0])))

    return df_actions_grid


if __name__ == "__main__":
    logging.basicConfig(filename=r"..\output\app.log", level=logging.DEBUG)
    matches_ids = get_matches_ids(os.path.join(STATSBOMB_OPEN_DATA_LOCAL_PATH, "data", "events"))
    start_time = time.time()
    df_events = load_events(matches_ids, parallel_processes_count=4)
    end_time = time.time()
    print(f"Load events elapsed time: {end_time-start_time}s")
    df_actions = get_actions(df_events)
    df_actions_grid = create_actions_grid(df_actions, FOOTBALL_PITCH_SIZE, HEATMAP_TILE_SIZE)
    df_actions_grid.to_csv(r"..\output\actions_grid.csv", index=False)

    print(df_actions_grid)