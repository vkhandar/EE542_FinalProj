import pandas as pd

# Load CSVs
players = pd.read_csv("data/2024allplayers.csv")    # retroID = id
people = pd.read_csv("data/People.csv")               # retroID, playerID
war = pd.read_csv("out/player_WAR_basic.csv")

# Normalize team IDs to modern names
team_normalize = {
    "ANA": "LAA",
    "CAL": "LAA",   # just in case
}

players["team"] = players["team"].replace(team_normalize)

war_latest = (
    war.sort_values(["playerID", "yearID"], ascending=[True, False])
       .drop_duplicates(subset="playerID", keep="first")
)


merged1 = players.merge(
    people,
    left_on="id",
    right_on="retroID",
    how="left"
)

merged2 = merged1.merge(
    war_latest[["playerID", "WAR_basic_total"]],
    on="playerID",
    how="left"
)

final = merged2[[
    "playerID",
    "nameLast",
    "nameFirst",
    "team",              # rename to teamID
    "WAR_basic_total"
]].rename(columns={"team": "teamID"})

final = final.sort_values(
    by=["teamID", "WAR_basic_total"],
    ascending=[True, False]
)


final.to_csv("output_players_war.csv", index=False)

print("Done! Created output_players_war.csv")