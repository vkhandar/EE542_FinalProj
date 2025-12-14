import pandas as pd
from pathlib import Path

# === EDIT THESE IF YOUR PATHS ARE DIFFERENT ===
DATA_DIR = Path("/home/damrow/FinalProjP6/data")
OUT_PATH = DATA_DIR / "2025schedule.csv"

def main():
    teams_path = DATA_DIR / "Teams.csv"
    teams = pd.read_csv(teams_path)

    # Use only 2024 MLB teams (AL + NL)
    teams_2024 = teams[teams["yearID"] == 2024].copy()
    teams_2024 = teams_2024[teams_2024["lgID"].isin(["AL", "NL"])]

    team_ids = sorted(teams_2024["teamID"].unique())
    print(f"Found {len(team_ids)} MLB teams for 2024: {team_ids}")

    rows = []
    start_date = pd.Timestamp("2025-03-28")
    season_days = 180  # length of season window for dates
    idx = 0

    # 3-game home series for every home/away pair
    GAMES_PER_MATCHUP = 3

    for home in team_ids:
        for away in team_ids:
            if home == away:
                continue
            for g in range(GAMES_PER_MATCHUP):
                date = start_date + pd.Timedelta(days=idx % season_days)
                rows.append(
                    {
                        "date": date.date().isoformat(),
                        "homeTeam": home,
                        "awayTeam": away,
                    }
                )
                idx += 1

    sched = pd.DataFrame(rows)
    sched.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(sched)} games to {OUT_PATH}")

if __name__ == "__main__":
    main()
