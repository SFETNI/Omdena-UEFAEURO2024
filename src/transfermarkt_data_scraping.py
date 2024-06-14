import os.path

import bs4
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path


OUTPUT_DIR_PATH = os.path.join(os.path.dirname(__file__), "..", "output")
LIVE_MATCHES_FILE_PATH = os.path.join(OUTPUT_DIR_PATH, "live_matches.csv")

Path(OUTPUT_DIR_PATH).mkdir(parents=True, exist_ok=True)


def get_page_parser(url: str) -> bs4.BeautifulSoup:
    headers = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

    pageTree = requests.get(url, headers=headers)
    pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

    return pageSoup


def get_match_details(match_details_url: str) -> None:
    pageSoup = get_page_parser(match_details_url)

    div_lineups_row = pageSoup.select("main#tm-main > .row")[2]

    return None


def get_live_score(url: str) -> pd.DataFrame:
    pageSoup = get_page_parser(url)
    div_categories = pageSoup.find_all("div", {"class": "kategorie"})
    competitions = {}
    data = np.array([], dtype=[("competition", str), ("home_team", str),
                               ("away_team", str), ("home_team_goals", int),
                               ("away_team_goals", int)])
    df_matches = pd.DataFrame(data, columns=["competition", "home_team", "away_team",
                                             "home_team_goals", "away_team_goals"])

    for div_category in div_categories:
        new_match_data = {"competition": div_category.text.strip()}
        table_live_scores = div_category.find_next_sibling("table", attrs={"class": "livescore"})
        tr_matches = table_live_scores.find_all("tr", {"class": "begegnungZeile"})
        for tr_match in tr_matches:
            td_home_team = tr_match.find("td", attrs={"class": "club verein-heim"})
            td_away_team = tr_match.find("td", attrs={"class": "club away verein-gast"})
            td_result = tr_match.find("td", attrs={"class": "ergebnis"})
            span_result = td_result.find("span", attrs={"class": "matchresult"})
            home_team_goals = ""
            away_team_goals = ""
            if "finished" in span_result.attrs["class"]:
                home_team_goals, away_team_goals = span_result.text.strip().split(":")
            new_match_data["home_team"] = td_home_team.text.strip()
            new_match_data["away_team"] = td_away_team.text.strip()
            new_match_data["home_team_goals"] = home_team_goals
            new_match_data["away_team_goals"] = away_team_goals
            a_match_details = td_result.find("a")
            #match_details = get_match_details(a_match_details.attrs["href"])

            df_matches = df_matches._append(new_match_data, ignore_index=True)

    return df_matches


if __name__ == "__main__":
    url = "https://www.transfermarkt.com/ticker/index/live"
    df_matches = get_live_score(url)
    df_matches.to_csv(LIVE_MATCHES_FILE_PATH, index=False)

