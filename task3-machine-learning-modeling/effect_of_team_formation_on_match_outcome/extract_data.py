import os
import sys
import math
import queue
import logging
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
import multiprocessing as mp
from statsbombpy_local import sb
from datetime import datetime, timedelta
from multiprocessing import current_process
from concurrent.futures import ProcessPoolExecutor


OUTPUT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..", "output")
APP_LOG_PATH = os.path.join(OUTPUT_DIR_PATH, "app.log")
STATSBOMB_OPEN_DATA_LOCAL_PATH = os.environ["OPEN_DATA_REPO_PATH"]
DATA_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "data.csv")
logger = logging.getLogger(__name__)
pd.options.mode.copy_on_write = True


def get_matches_ids(data_path: str, max_events=-1) -> list[int]:
    matches_ids = []

    for idx, data_file in enumerate(os.listdir(data_path)):
        match_id = int(data_file.split(".")[0])
        matches_ids.append(match_id)
        if max_events != -1 and idx+1 == max_events:
            break

    return matches_ids


def update_nan_timestamp(row, match_end_timestamp):
    if row["formation_play_duration"] is pd.NaT:
        return match_end_timestamp - row["timestamp"]

    return row["formation_play_duration"]


def normalize_action_data(df_data: pd.DataFrame, match_end_timestamp: timedelta) -> pd.DataFrame:
    df_data_normalized = df_data.copy()
    # If the formation_play_duration_ration value is 1 it means that the formation is being played for the entire match.
    # The value of 2 means that the formation is played either during the first or the second half of the match.
    df_data_normalized["formation_play_duration_ratio"] = ((match_end_timestamp-datetime(1900, 1, 1)).total_seconds() /
                                                            df_data["formation_play_duration"].dt.total_seconds())
    df_data_normalized["shot_mean"] = df_data["shot"] * df_data_normalized["formation_play_duration_ratio"]
    df_data_normalized["pass_mean"] = df_data["pass"] * df_data_normalized["formation_play_duration_ratio"]
    df_data_normalized["under_mean"] = df_data["under_pressure"] * df_data_normalized["formation_play_duration_ratio"]
    df_data_normalized["counterpress_mean"] = df_data["counterpress"] * df_data_normalized["formation_play_duration_ratio"]

    return df_data_normalized


def filter_actions(df_events: pd.DataFrame, df_tactics: pd.DataFrame, match_id: int) -> pd.DataFrame:
    actions_filter = ((df_events.under_pressure == True) | (df_events.counterpress == True) |
                      (df_events.type == "Shot") | (df_events.type == "Pass"))
    df_selected_actions = df_events.loc[actions_filter]
    df_selected_actions["timestamp"] = pd.to_datetime(df_selected_actions["timestamp"], format='%H:%M:%S.%f')
    for index, row in df_tactics.iterrows():
        df_selected_actions.loc[((df_selected_actions["team"] == row["team"]) &
                                 (df_selected_actions.timestamp > row["timestamp"])), "formation"] = row["formation"]
    df_selected_actions["pass"] = 0
    df_selected_actions["shot"] = 0
    df_selected_actions["score"] = 0
    df_selected_actions.loc[df_selected_actions.type == "Pass", "pass"] = 1
    df_selected_actions.loc[df_selected_actions.type == "Shot", "shot"] = 1
    df_selected_actions.loc[df_selected_actions.shot_outcome == "Goal", "score"] = 1
    df_selected_actions["counterpress"] = df_selected_actions["counterpress"].fillna(0)
    df_selected_actions["under_pressure"] = df_selected_actions["under_pressure"].fillna(0)
    df_selected_actions.loc[df_selected_actions.counterpress == True, "counterpress"] = 1
    df_selected_actions["match_id"] = match_id

    return df_selected_actions


def get_match_end_timestamp(df_events: pd.DataFrame, period: int = 1) -> timedelta:
    str_half_end_timestamp = df_events.loc[(df_events.type == "Half End") & (df_events.period == 1)]["timestamp"].tolist()[0]
    half_end_timestamp = datetime.strptime(str_half_end_timestamp, "%H:%M:%S.%f")
    if period == 1:
        return half_end_timestamp
    half_end_timestamp_delta = timedelta(minutes=half_end_timestamp.minute,
                                         seconds=half_end_timestamp.second,
                                         microseconds=half_end_timestamp.microsecond)
    str_match_end_timestamp = df_events.loc[(df_events.type == "Half End") & (df_events.period == 2)]["timestamp"].tolist()[0]
    match_end_timestamp = datetime.strptime(str_match_end_timestamp, "%H:%M:%S.%f")
    match_end_total_timestamp = match_end_timestamp + half_end_timestamp_delta

    return match_end_total_timestamp


def get_match_tactics(df_events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    tactics_filter = (df_events.type == "Starting XI") | (df_events.type == "Tactical Shift")
    df_tactics = df_events.loc[tactics_filter][["team", "tactics", "timestamp", "period"]]
    df_tactics.reset_index(drop=True, inplace=True)
    df_tactics["timestamp"] = pd.to_datetime(df_tactics["timestamp"], format="%H:%M:%S.%f")
    # Change timestamp to range from 0 to 90 minutes.
    half_end_timestamp = get_match_end_timestamp(df_events)
    df_tactics.loc[df_tactics.period == 2, "timestamp"] += timedelta(minutes=half_end_timestamp.minute,
                                                                     seconds=half_end_timestamp.second,
                                                                     microseconds=half_end_timestamp.microsecond)
    df_red_card_events = df_events.loc[
        ((df_events.type == "Bad Behaviour") & (df_events.get("bad_behaviour_card") == "Red Card")) |
        ((df_events.type == "Foul Committed") & (df_events.get("foul_committed_card") == "Red Card"))
    ]
    if df_red_card_events is not None and len(df_red_card_events.index) > 0:
        logger.debug(f"There are {len(df_red_card_events.index)} red cards in the match with in ID {match_id}.")
        # Select just the first red card event. We are not interested in the rest of the events because the
        # first red card event will mark the end of the match.
        first_red_card_event_timestamp = datetime.strptime(df_red_card_events.iloc[0]["timestamp"], "%H:%M:%S.%f")
        df_tactics_filtered = df_tactics.loc[df_tactics["timestamp"] < first_red_card_event_timestamp]
        logger.debug(f"Number of tactics change events prior to the red card event: {len(df_tactics_filtered)}")
        logger.debug(f"Total number of tactics change events: {len(df_tactics)}")
        df_tactics = df_tactics_filtered
        if df_tactics is None or len(df_tactics) == 0:
            return None
        # Match events collection should end with the first red card event.
        match_end_timestamp = first_red_card_event_timestamp
    df_tactics["formation"] = df_tactics["tactics"].apply(lambda row: row["formation"]).astype(str)
    df_tactics["diffs"] = df_tactics.groupby(by=["team"])["timestamp"].diff()
    # Shift values in colums with timestamp differences up one row
    df_tactics["formation_play_duration"] = df_tactics.groupby(by=["team"])["diffs"].transform(lambda r: r.shift(-1))
    match_end_timestamp = get_match_end_timestamp(df_events, period=2)
    df_tactics["formation_play_duration"] = df_tactics.apply(update_nan_timestamp, args=(match_end_timestamp,), axis=1)
    df_tactics.drop(columns=["tactics", "diffs", "period"], inplace=True)

    return df_tactics


def extract_data_worker(matches_queue: queue.Queue, fetched_data_queue: queue.Queue) -> None:
    pd.options.mode.copy_on_write = True
    matches_ids = matches_queue.get()
    df_data = None

    logger = logging.getLogger()
    # log all messages, debug and up
    logger.setLevel(logging.DEBUG)
    process = current_process()
    logger.debug(f"{process.name} | matches size {len(matches_ids)}")

    for match_id in matches_ids:
        df_events = sb.events(match_id=match_id)
        df_tactics = get_match_tactics(df_events, match_id)
        # The tactics DataFrame can be empty in case of red card event that happend early in the match.
        # In that case skip processing the match.
        if df_tactics is None or len(df_tactics) == 0:
            continue
        df_selected_actions = filter_actions(df_events, df_tactics, match_id)
        df_tactics_groupped = df_tactics[["team", "formation", "formation_play_duration"]].groupby(
            by=["team", "formation"]).sum().reset_index()
        df_actions_groupped = df_selected_actions[
            ["match_id", "team", "formation", "shot", "pass", "under_pressure", "score", "counterpress"]].groupby(
            by=["match_id", "team", "formation"]).sum().reset_index()
        df_actions_groupped = df_actions_groupped.merge(df_tactics_groupped, how="inner", on=["team", "formation"])
        match_end_timestamp = get_match_end_timestamp(df_events, period=2)
        df_actions_groupped = normalize_action_data(df_actions_groupped, match_end_timestamp)
        home_team = df_events.iloc[0]["team"]
        away_team = df_events.iloc[1]["team"]
        df_actions_groupped["home_or_away"] = 1
        df_actions_groupped.loc[df_actions_groupped["team"] == away_team, "home_or_away"] = 0
        df_data = df_actions_groupped.copy() if df_data is None else pd.concat([df_data, df_actions_groupped])

    mean_expected_goals = df_data["goals"].sum() / len(df_data)
    df_data["mean_expected_goals_per_game"] = mean_expected_goals
    df_data["goals_diffs_from_mean"] = df_data["goals"] - df_data["mean_expected_goals_per_game"]
    df_data["variance_of_expected_goals_per_game"] = df_data["goals_diffs_from_mean"].pow(2) / len(df_data)
    df_data.drop(columns=["goals_diffs_from_mean"], inplace=True)

    fetched_data_queue.put(df_data)
    matches_queue.task_done()


def extract_data_single_proc(matches_ids: list[int]) -> pd.DataFrame:
    df_data = None

    for match_id in matches_ids:
        df_events = sb.events(match_id=match_id)
        df_tactics = get_match_tactics(df_events, match_id)
        # The tactics DataFrame can be empty in case of red card event that happend early in the match.
        # In that case skip processing the match.
        if df_tactics is None or len(df_tactics) == 0:
            continue
        df_selected_actions = filter_actions(df_events, df_tactics, match_id)
        df_tactics_groupped = df_tactics[["team", "formation", "formation_play_duration"]].groupby(
            by=["team", "formation"]).sum().reset_index()
        df_actions_groupped = df_selected_actions[
            ["match_id", "team", "formation", "shot", "pass", "under_pressure", "score", "counterpress"]].groupby(
            by=["match_id", "team", "formation"]).sum().reset_index()
        df_actions_groupped = df_actions_groupped.merge(df_tactics_groupped, how="inner", on=["team", "formation"])
        match_end_timestamp = get_match_end_timestamp(df_events, period=2)
        df_actions_groupped = normalize_action_data(df_actions_groupped, match_end_timestamp)
        home_team = df_events.iloc[0]["team"]
        away_team = df_events.iloc[1]["team"]
        df_actions_groupped["home_or_away"] = 1
        df_actions_groupped.loc[df_actions_groupped["team"] == away_team, "home_or_away"] = 0
        df_goals = df_events.loc[df_events.shot_outcome == "Goal"]
        df_actions_groupped["goals"] = len(df_goals)
        df_data = df_actions_groupped.copy() if df_data is None else pd.concat([df_data, df_actions_groupped])

    mean_expected_goals = df_data["goals"].sum()/len(df_data)
    df_data["mean_expected_goals_per_game"] = mean_expected_goals
    df_data["goals_diffs_from_mean"] = df_data["goals"] - df_data["mean_expected_goals_per_game"]
    df_data["variance_of_expected_goals_per_game"] = df_data["goals_diffs_from_mean"].pow(2) / len(df_data)
    df_data.drop(columns=["goals_diffs_from_mean"], inplace=True)

    return df_data


def extract_data(matches_ids: list[int], parallel_processes_count=os.cpu_count()) -> pd.DataFrame:
    matches_count = len(matches_ids)
    matches_slice_size = round(len(matches_ids) / parallel_processes_count)

    with mp.Manager() as manager:
        fetched_data_queue = manager.Queue()
        matches_queue = manager.Queue()
        for i in range(parallel_processes_count):
            start = i * matches_slice_size
            end = (i + 1) * matches_slice_size
            logger.debug(f"Start slice: {start}, end slice: {end}")
            matches_queue.put(matches_ids[start:end])
        logger.debug(f"Queue size: {matches_queue.qsize()}")
        with ProcessPoolExecutor(max_workers=parallel_processes_count) as executor:
            executor.map(extract_data_worker,
                         [matches_queue]*parallel_processes_count,
                         [fetched_data_queue]*parallel_processes_count)
        matches_queue.join()
        logger.debug(f"Fecthed data queue size: {fetched_data_queue.qsize()}")
        df_data = None
        while not fetched_data_queue.empty():
            df_temp = fetched_data_queue.get()
            df_data = df_temp if df_data is None else pd.concat([df_data, df_temp])

    return df_data


if __name__ == "__main__":
    start_time = time()
    Path(OUTPUT_DIR_PATH).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler(APP_LOG_PATH), logging.StreamHandler(sys.stdout)])
    matches_ids = get_matches_ids(os.path.join(STATSBOMB_OPEN_DATA_LOCAL_PATH, "data", "events"))
    logger.debug(f"Matches count: {len(matches_ids)}")
    #df_data = extract_data(matches_ids, parallel_processes_count=6)
    df_data = extract_data_single_proc(matches_ids)
    df_data.sort_values(by="match_id", inplace=True)
    df_data.to_csv(DATA_FILE_PATH, index=False)
    elapsed_time = time() - start_time
    logger.debug(f"Elapsed time: {elapsed_time}")
    print(len(df_data))