import os.path

import bs4
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin


COMPETITION_NAME = "INTERNATIONAL FRIENDLIES"
OUTPUT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..", "output")
LIVE_MATCHES_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "live_matches.csv")
LINEUPS_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "lineups.csv")

Path(OUTPUT_DIR_PATH).mkdir(parents=True, exist_ok=True)


def get_page_parser(url: str) -> bs4.BeautifulSoup:
    headers = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

    pageTree = requests.get(url, headers=headers)
    pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

    return pageSoup


def get_player_profile(profile_url: str) -> dict:
    player_profile = {}
    pageSoup = get_page_parser(profile_url)
    div_player_name = pageSoup.select_one(".data-header__headline-container")
    player_name = div_player_name.text.strip()
    # Strip player number and superfluous blank spaces from player name
    player_name = " ".join(player_name.split(" ")[-2:])
    div_player_info = pageSoup.select(".data-header__info-box .data-header__items")[1]
    player_pos = None
    for li_player_data in div_player_info.select(".data-header__label"):
        text = li_player_data.text.strip()
        if "Position" in text:
            player_pos = text.split(" ")[-1].strip()
            break
    player_profile["player_position"] = get_play_position_code(player_pos)
    player_profile["player_name"] = player_name

    return player_profile


def get_play_position_code(player_position: str) -> str:
    player_pos_codes = {
        "Goalkeeper": "GK",
        "Center-Back": "CB",
        "Centre-Back": "CB",
        "Right-Back": "RB",
        "Left-Back": "LB",
        "Left Winger": "LW",
        "Right Winger": "RW",
        "Defender": "D",
        "Attack": "ST",
        "Center-Forward": "CF",
        "Centre-Forward": "CF",
        "Central Midfield": "CM",
        "Defensive Midfield": "DM",
        "Attacking Midfield": "AM",
        "Midfield": "M",
        "Winger": "W"
    }

    return player_pos_codes[player_position]


def get_team_lineup(lineup_elem: bs4.element.Tag, match_report_url: str) -> dict:
    lineup = {"team_tactics": "", "players": []}
    div_team_name = lineup_elem.find("div", attrs={"class": "aufstellung-unterueberschrift-mannschaft"})
    lineup["team_name"] = div_team_name.text.strip()
    div_row = div_team_name.find_next_sibling("div", attrs={"class": "row"})
    if not div_row:
        div_columns = div_team_name.find_next_sibling()
        tr_lineup_rows = div_columns.select("table > tr")
        player_pos, player_name = [cell.text.strip() for cell in tr_lineup_rows.find_all("td")]
        if player_pos.endswith("s"):
            player_pos = player_pos[:-1]
        player_pos_code = get_play_position_code(player_pos)
        lineup["players"].append({ "player_position": player_pos_code, "player_name": player_name })
    else:
        div_team_tactics = div_row.select_one("div.aufstellung-vereinsseite")
        lineup["team_tactics"] = div_team_tactics.text.strip()
        div_player_containers = lineup_elem.select("div.aufstellung-spieler-container")
        for div_player_container in div_player_containers:
            span_player_name = div_player_container.select_one("span.aufstellung-rueckennummer-name")
            player_name = span_player_name.text.strip()
            a_player_profil_url = span_player_name.find("a")
            player_profil_url = a_player_profil_url.get("href")
            player_profile = get_player_profile(urljoin(match_report_url, player_profil_url))
            lineup["players"].append({ "player_position": player_profile["player_position"], "player_name": player_name })

    return lineup


def get_match_report(match_report_url: str) -> None:
    pageSoup = get_page_parser(match_report_url)

    div_lineups_row = pageSoup.select("main#tm-main > .row")[2]
    div_lineups_columns = div_lineups_row.select(".box > .columns")

    home_team_lineup = get_team_lineup(div_lineups_columns[0], match_report_url)
    away_team_lineup = get_team_lineup(div_lineups_columns[1], match_report_url)

    match_report = {
        "home_team_report": { "lineup": home_team_lineup },
        "away_team_report": { "lineup": away_team_lineup }
    }

    return match_report


def add_team_lineup(team_report: dict, df_lineups: pd.DataFrame) -> None:
    df_temp = df_lineups

    for player in team_report["lineup"]["players"]:
        data = {"team_name": team_report["lineup"]["team_name"], "team_tactics": team_report["lineup"]["team_tactics"]}
        data["player_name"] = player["player_name"]
        data["player_position"] = player["player_position"]
        df_temp = df_temp._append(data, ignore_index=True)

    return df_temp


def get_live_score(url: str, competitions_filter: list[str]) -> pd.DataFrame:
    pageSoup = get_page_parser(url)
    div_categories = pageSoup.find_all("div", {"class": "kategorie"})
    competitions = {}
    matches_data = np.array([], dtype=[("competition", str), ("home_team", str),
                               ("away_team", str), ("home_team_goals", int),
                               ("away_team_goals", int)])
    lineups_data = np.array([], dtype=[("team_name", str), ("player_name", str), ("player_pos", str)])
    df_matches = pd.DataFrame(matches_data, columns=["competition", "home_team", "away_team",
                                             "home_team_goals", "away_team_goals"])
    df_lineups = pd.DataFrame(lineups_data, columns=["team_name", "player_name", "player_pos"])

    for div_category in div_categories:
        competition_category_name = div_category.text.strip()
        if competition_category_name.lower() in competitions_filter:
            new_match_data = {"competition": div_category.text.strip()}
            table_live_scores = div_category.find_next_sibling("table", attrs={"class": "livescore"})
            tr_matches = table_live_scores.find_all("tr", {"class": "begegnungZeile"})
            for tr_match in tr_matches:
                td_home_team = tr_match.find("td", attrs={"class": "club verein-heim"})
                td_away_team = tr_match.find("td", attrs={"class": "club away verein-gast"})
                home_team = td_home_team.text.strip()
                away_team = td_away_team.text.strip()
                td_result = tr_match.find("td", attrs={"class": "ergebnis"})
                span_result = td_result.find("span", attrs={"class": "matchresult"})
                home_team_goals = ""
                away_team_goals = ""
                if any(i in span_result.attrs["class"] for i in ["finished"]):
                    home_team_goals, away_team_goals = span_result.text.strip().split(":")
                    a_match_report = td_result.find("a")
                    match_report_url = urljoin(url, a_match_report.get("href"))
                    try:
                        match_report = get_match_report(match_report_url)
                        df_lineups = add_team_lineup(match_report["home_team_report"], df_lineups)
                        df_lineups = add_team_lineup(match_report["away_team_report"], df_lineups)
                    except Exception as ex:
                        print(f"Failed to process data for match: {home_team} vs {away_team}")
                new_match_data["home_team"] = home_team
                new_match_data["away_team"] = away_team
                new_match_data["home_team_goals"] = home_team_goals
                new_match_data["away_team_goals"] = away_team_goals
                a_match_details = td_result.find("a")
                #match_details = get_match_details(a_match_details.attrs["href"])

                df_matches = df_matches._append(new_match_data, ignore_index=True)

    return df_matches, df_lineups


if __name__ == "__main__":
    url = "https://www.transfermarkt.com/ticker/index/live"
    df_matches, df_lineups = get_live_score(url, competitions_filter=[COMPETITION_NAME.lower()])
    df_matches.to_csv(LIVE_MATCHES_FILE_PATH, index=False)
    df_lineups.to_csv(LINEUPS_FILE_PATH, index=False)
