import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply.*", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans

try:
    import xgboost as xgb
    USE_XGB = True
    print("[XGBOOST] Using XGBoost GPU (tree_method='gpu_hist').")
except ImportError:
    USE_XGB = False
    print("[XGBOOST] xgboost not installed, falling back to sklearn GradientBoosting.")
    from sklearn.ensemble import GradientBoostingClassifier

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_OK = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TORCH] Using device: {DEVICE}")
except Exception:
    TORCH_OK = False
    DEVICE = None


SEED = 42
np.random.seed(SEED)


N_CLUSTERS = 8


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(a, dtype=float)
    np.divide(a, b, out=out, where=b !=0)
    return out

def get_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    return (df[name] if name in df.columns else pd.Series(default, index=df.index)).fillna(0)

def minmax_series(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    v = s.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    try:
        scaled = scaler.fit_transform(v).ravel()
    except Exception:
        scaled = np.zeros_like(v.ravel(), dtype=float)
    return pd.Series(scaled, index=s.index)

def load_csvs(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    people = pd.read_csv(data_dir / "People.csv")
    bat = pd.read_csv(data_dir / "Batting.csv")
    pit = pd.read_csv(data_dir / "Pitching.csv")
    fld = pd.read_csv(data_dir / "Fielding.csv")
    return people, bat, pit, fld

def compute_basic_batting_war(bat: pd.DataFrame) -> pd.DataFrame:

    df = bat.copy()


    for col in ["BB", "HBP", "SF"]:
        if col not in df.columns:
            df[col] = 0
    df["PA"] = df["AB"] + df["BB"] + df["HBP"] + df["SF"]

    # Avoid div-by-zero
    df = df[df["PA"] > 0].copy()

    # Compute OBP and SLG
    # Need: H, 2B, 3B, HR, BB, HBP, SF
    for col in ["H", "2B", "3B", "HR"]:
        if col not in df.columns:
            df[col] = 0

    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    TB = singles + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]

    df["OBP"] = (df["H"] + df["BB"] + df["HBP"]) / (df["AB"] + df["BB"] + df["HBP"] + df["SF"])
    df["SLG"] = TB / df["AB"].replace(0, np.nan)
    df["OBP"] = df["OBP"].fillna(0.0)
    df["SLG"] = df["SLG"].fillna(0.0)

    # League averages per year
    lg = df.groupby("yearID").agg(
        lg_OBP=("OBP", "mean"),
        lg_SLG=("SLG", "mean"),
    ).reset_index()

    df = df.merge(lg, on="yearID", how="left")


    k = 1.5
    df["Bat_Runs_AA"] = (df["OBP"] + df["SLG"] - df["lg_OBP"] - df["lg_SLG"]) * df["PA"] * k


    df["Bat_Runs_RAR"] = df["Bat_Runs_AA"] - (-20.0 * df["PA"] / 600.0)

    df["Bat_WAR_basic"] = df["Bat_Runs_RAR"] / 10.0

    return df[["playerID", "yearID", "Bat_WAR_basic", "Bat_Runs_RAR", "PA"]]

def compute_basic_pitching_war(pit: pd.DataFrame) -> pd.DataFrame:
    df = pit.copy()

    # Require IPouts, R
    for col in ["IPouts", "R"]:
        if col not in df.columns:
            df[col] = 0

    df["IP"] = df["IPouts"] / 3.0
    df = df[df["IP"] > 0].copy()

    # RA9 = 9 * R / IP
    df["RA9"] = 9.0 * df["R"] / df["IP"]


    lg = df.groupby("yearID").agg(
        lg_R=("R", "sum"),
        lg_IP=("IP", "sum"),
    ).reset_index()
    lg["lg_RA9"] = 9.0 * lg["lg_R"] / lg["lg_IP"]

    df = df.merge(lg[["yearID", "lg_RA9"]], on="yearID", how="left")

    # Runs Above Average: (lg_RA9 - RA9) * IP / 9
    df["Pit_Runs_AA"] = (df["lg_RA9"] - df["RA9"]) * df["IP"] / 9.0


    df["Pit_Runs_RAR"] = df["Pit_Runs_AA"] - (-0.38 * df["IP"] / 9.0)

    # Convert runs to wins
    df["Pit_WAR_basic"] = df["Pit_Runs_RAR"] / 10.0

    return df[["playerID", "yearID", "Pit_WAR_basic", "Pit_Runs_RAR", "IP"]]

def load_yearly_teamstats(
    data_dir: Path,
    start_year: int = 2014,
    end_year: int = 2024,
) -> pd.DataFrame:

    dfs = []
    for year in range(start_year, end_year + 1):
        path = data_dir / f"{year}teamstats.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["yearID"] = year
            dfs.append(df)
        else:
            print(f"[WARN] {path.name} not found, skipping that year.")

    if not dfs:
        raise FileNotFoundError("No YYYYteamstats.csv files found in data_dir.")

    ts_all = pd.concat(dfs, ignore_index=True)
    print(f"[YEARLY] Loaded {len(ts_all)} rows from {start_year}-{end_year} teamstats.")
    return ts_all

def load_yearly_gameinfo(
    data_dir: Path,
    start_year: int = 2014,
    end_year: int = 2024,
):

    gi_list = []
    for year in range(start_year, end_year + 1):
        path = data_dir / f"{year}gameinfo.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["yearID"] = year
            gi_list.append(df)
        else:
            print(f"[WARN] {path.name} not found, skipping gameinfo for {year}.")
    if gi_list:
        gi_all = pd.concat(gi_list, ignore_index=True)
        print(f"[YEARLY] Loaded {len(gi_all)} rows of gameinfo.")
        return gi_all
    return pd.DataFrame()

def build_games_with_enhancements(ts: pd.DataFrame, gi: pd.DataFrame):

    ts = ts.copy()

    ts["date"] = pd.to_datetime(ts["gid"].str[3:11], errors="coerce")
    ts = ts.dropna(subset=["date"])

    # Rolling stats per team
    ts = ts.sort_values(["team", "date"])

    def add_team_roll(g):
        g = g.sort_values("date")
        runs = g["b_r"].fillna(0)

        g["roll_runs_5"] = runs.rolling(5, min_periods=1).mean().shift(1)
        g["roll_runs_10"] = runs.rolling(10, min_periods=1).mean().shift(1)

        wins = (g["b_r"] > g["b_r"].shift(1).fillna(0)).astype(int)
        g["roll_winpct_10"] = wins.rolling(10, min_periods=1).mean().shift(1)
        return g.fillna(0)

    ts = ts.groupby("team", group_keys=False).apply(add_team_roll)

    # Merge home/away rows
    home = ts[ts["vishome"] == "h"]
    away = ts[ts["vishome"] == "v"]
    merged = home.merge(away, on="gid", suffixes=("_home", "_away"))

    games = pd.DataFrame({
        "gid": merged["gid"],
        "date": merged["date_home"],
        "yearID": merged["date_home"].dt.year,
        "homeTeam": merged["team_home"],
        "awayTeam": merged["team_away"],
        "homeScore": merged["b_r_home"],
        "awayScore": merged["b_r_away"],
        "home_roll_runs_5": merged["roll_runs_5_home"],
        "home_roll_runs_10": merged["roll_runs_10_home"],
        "home_roll_winpct_10": merged["roll_winpct_10_home"],
        "away_roll_runs_5": merged["roll_runs_5_away"],
        "away_roll_runs_10": merged["roll_runs_10_away"],
        "away_roll_winpct_10": merged["roll_winpct_10_away"],
    })

    if not gi.empty and "gid" in gi.columns:
        keep = ["gid"] + [c for c in ["temp", "wind", "humidity", "windspeed"] if c in gi.columns]
        gi_small = gi[keep].drop_duplicates(subset=["gid"], keep="last")
        games = games.merge(gi_small, on="gid", how="left")

        for col in ["temp", "wind", "humidity", "windspeed"]:
            if col in games.columns:
                games[col] = games[col].astype(float).fillna(games[col].mean())

    return games.reset_index(drop=True)

def build_games_from_teamstats(ts: pd.DataFrame) -> pd.DataFrame:

    ts = ts.copy()

    ts = ts[ts["vishome"].isin(["h", "v"])].copy()

    home = ts[ts["vishome"] == "h"].copy()
    away = ts[ts["vishome"] == "v"].copy()

    merged = home.merge(
        away,
        on="gid",
        suffixes=("_home", "_away"),
        how="inner",
    )

    merged["date"] = pd.to_datetime(
        merged["gid"].str[3:11],  # YYYYMMDD
        format="%Y%m%d",
        errors="coerce",
    )
    merged["yearID"] = merged["date"].dt.year

    games = pd.DataFrame(
        {
            "date": merged["date"],
            "yearID": merged["yearID"],
            "homeTeam": merged["team_home"],
            "awayTeam": merged["team_away"],
            "homeScore": merged["b_r_home"],
            "awayScore": merged["b_r_away"],
        }
    )


    games = games.dropna(subset=["date"]).reset_index(drop=True)
    print(f"[GAMES] Built {len(games)} games from teamstats.")
    return games

def build_player_year(people: pd.DataFrame, bat: pd.DataFrame, pit: pd.DataFrame, fld: pd.DataFrame) -> pd.DataFrame:
    #  Batting features
    bat_ps = bat.copy()
    for c in ["H","AB","BB","HBP","SF","2B","3B","HR","RBI","SO","G"]:
        bat_ps[c] = get_col(bat_ps, c)
    bat_ps["1B"] = (bat_ps["H"] - bat_ps["2B"] - bat_ps["3B"] - bat_ps["HR"]).clip(lower=0)
    bat_ps["AVG"] = safe_div(bat_ps["H"], bat_ps["AB"])
    bat_ps["OBP"] = safe_div(bat_ps["H"] + bat_ps["BB"] + bat_ps["HBP"],
                              bat_ps["AB"] + bat_ps["BB"] + bat_ps["HBP"] + bat_ps["SF"])
    tb = (1*bat_ps["1B"] + 2*bat_ps["2B"] + 3*bat_ps["3B"] + 4*bat_ps["HR"])
    bat_ps["SLG"] = safe_div(tb, bat_ps["AB"])
    bat_ps["OPS"] = (bat_ps["OBP"] + bat_ps["SLG"]).fillna(0)
    bat_feats = bat_ps.groupby(["playerID","yearID"], as_index=False).agg(
        AB=("AB","sum"), H=("H","sum"), HR=("HR","sum"), RBI=("RBI","sum"),
        BB=("BB","sum"), SO=("SO","sum"), HBP=("HBP","sum"), SF=("SF","sum"),
        G=("G","sum"),
        AVG=("AVG","mean"), OBP=("OBP","mean"), SLG=("SLG","mean"), OPS=("OPS","mean")
    )

    #  Pitching features
    pit_ps = pit.copy()
    # Lahman uses 'IPouts' (lowercase o)
    ip_col = "IPouts" if "IPouts" in pit_ps.columns else ("IPOuts" if "IPOuts" in pit_ps.columns else None)
    if ip_col is None:
        pit_ps["IP"] = 0.0
    else:
        pit_ps["IP"] = safe_div(get_col(pit_ps, ip_col), 3.0)

    for c in ["ER","BB","H","SO","R","W","L","SV","ERA"]:
        pit_ps[c] = get_col(pit_ps, c)

    pit_ps["ERA_calc"] = np.where(pit_ps["IP"] > 0, 9.0 * pit_ps["ER"] / pit_ps["IP"], np.nan)
    pit_ps["ERA_use"] = pit_ps["ERA"].replace(0, np.nan).fillna(pit_ps["ERA_calc"])
    pit_ps["WHIP"] = np.where(pit_ps["IP"] > 0, (pit_ps["BB"] + pit_ps["H"]) / pit_ps["IP"], np.nan)

    pit_feats = pit_ps.groupby(["playerID","yearID"], as_index=False).agg(
        IP=("IP","sum"), SO=("SO","sum"), BB=("BB","sum"), H=("H","sum"),
        ER=("ER","sum"), R=("R","sum"), ERA=("ERA_use","mean"), WHIP=("WHIP","mean"),
        W=("W","sum"), L=("L","sum"), SV=("SV","sum")
    )

    #  Fielding features
    fld_ps = fld.copy()
    for c in ["PO","A","E","G"]:
        fld_ps[c] = get_col(fld_ps, c)
    fld_ps["TC"] = fld_ps["PO"] + fld_ps["A"] + fld_ps["E"]
    fld_ps["FPCT"] = np.where(fld_ps["TC"] > 0, (fld_ps["PO"] + fld_ps["A"]) / fld_ps["TC"], np.nan)
    fld_ps["RF"] = np.where(fld_ps["G"] > 0, (fld_ps["PO"] + fld_ps["A"]) / fld_ps["G"], np.nan)

    fld_feats = fld_ps.groupby(["playerID","yearID"], as_index=False).agg(
        PO=("PO","sum"), A=("A","sum"), E=("E","sum"), G=("G","sum"),
        FPCT=("FPCT","mean"), RF=("RF","mean")
    )

    #Merge
    plyr = bat_feats.merge(pit_feats, on=["playerID","yearID"], how="outer")
    plyr = plyr.merge(fld_feats, on=["playerID","yearID"], how="outer")
    plyr = plyr.fillna(0)

    # Attach names
    names = people[["playerID","nameFirst","nameLast"]].copy()
    names["playerName"] = (names["nameFirst"].fillna("") + " " + names["nameLast"].fillna("")).str.strip()
    plyr = plyr.merge(names[["playerID","playerName"]], on="playerID", how="left")

    return plyr

def get_series(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(0.0, index=df.index)

def build_subscores_per_year(plyr: pd.DataFrame) -> pd.DataFrame:
    def build_subscores(df_year: pd.DataFrame) -> pd.DataFrame:
        out = df_year.copy()

        #  Batting subscore
        bat_components = []
        for stat in ["OPS","OBP","SLG","HR","RBI","H"]:
            if stat in out.columns:
                bat_components.append(minmax_series(out[stat]))
        out["BattingScore"] = np.mean(bat_components, axis=0) if bat_components else 0.0

        #  Pitching subscore
        pit_components = []
        if "SO"   in out.columns: pit_components.append(minmax_series(out["SO"]))
        if "ERA"  in out.columns: pit_components.append(1.0 - minmax_series(out["ERA"]))
        if "WHIP" in out.columns: pit_components.append(1.0 - minmax_series(out["WHIP"]))
        out["PitchingScore"] = np.mean(pit_components, axis=0) if pit_components else 0.0

        #  Fielding subscore
        fld_components = []
        if "FPCT" in out.columns: fld_components.append(minmax_series(out["FPCT"]))
        if "RF"   in out.columns: fld_components.append(minmax_series(out["RF"]))
        if "E"    in out.columns: fld_components.append(1.0 - minmax_series(out["E"]))
        out["FieldingScore"] = np.mean(fld_components, axis=0) if fld_components else 0.0


        PA = (get_series(out, "AB") + get_series(out, "BB") +
        get_series(out, "HBP") + get_series(out, "SF")).fillna(0).to_numpy()
        IP = get_series(out, "IP").fillna(0).to_numpy()
        TC = (get_series(out, "PO") + get_series(out, "A") +
        get_series(out, "E")).fillna(0).to_numpy()
        Gf = get_series(out, "G").fillna(0).to_numpy()

        has_bat = (PA > 0)
        has_pit = (IP > 0)
        has_fld = (TC > 0) | (Gf > 0)
        bat_eff = np.where(has_bat, out["BattingScore"].to_numpy(float), 0.0)
        pit_eff = np.where(has_pit, out["PitchingScore"].to_numpy(float), 0.0)
        fld_eff = np.where(has_fld, out["FieldingScore"].to_numpy(float), 0.0)

        scores = out[["BattingScore","PitchingScore","FieldingScore"]].to_numpy(dtype=float)
        base_w = np.ones(3, dtype=float)

        avail = np.vstack([has_bat, has_pit, has_fld]).T.astype(float)
        wrow  = avail * base_w
        denom = wrow.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        wrow = wrow / denom

        out["OverallScore"] = bat_eff + pit_eff + fld_eff
        return out

    gb = plyr.groupby("yearID", group_keys=False)
    try:
        ranked = gb.apply(
            lambda g: build_subscores(g).assign(yearID=g.name),
            include_groups=False
        ).reset_index(drop=True)
    except TypeError:
        ranked = gb.apply(
            lambda g: build_subscores(g.drop(columns=["yearID"], errors="ignore")).assign(yearID=g.name)
        ).reset_index(drop=True)

    ranked["rank_in_year"] = ranked.groupby("yearID")["OverallScore"].rank(ascending=False, method="min")
    return ranked

def build_career(ranked: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    # Build playing time weights: PA + 3*IP
    pit_ps = pit.copy()
    ip_col = "IPouts" if "IPouts" in pit_ps.columns else ("IPOuts" if "IPOuts" in pit_ps.columns else None)
    if ip_col is None:
        pit_ps["IP"] = 0.0
    else:
        pit_ps["IP"] = safe_div(get_col(pit_ps, ip_col), 3.0)

    pit_ip_by_year = pit_ps.groupby(["playerID","yearID"])["IP"].sum().reset_index()
    key = ranked[["playerID","yearID"]].copy()
    key = key.merge(pit_ip_by_year, on=["playerID","yearID"], how="left").fillna({"IP":0.0})

    PA = (ranked.get("AB", 0) + ranked.get("BB", 0) + ranked.get("HBP", 0) + ranked.get("SF", 0)).fillna(0).values
    IP = key["IP"].values
    play_weight = (PA + 3.0 * IP)
    play_weight = np.where(play_weight == 0, 1.0, play_weight)
    ranked = ranked.copy()
    ranked["play_weight"] = play_weight

    career = (
        ranked.groupby("playerID", as_index=False).apply(
            lambda g: pd.Series({
                "career_OverallScore": np.average(g["OverallScore"], weights=g["play_weight"]),
                "career_BattingScore": np.average(g["BattingScore"], weights=g["play_weight"]),
                "career_PitchingScore": np.average(g["PitchingScore"], weights=g["play_weight"]),
                "career_FieldingScore": np.average(g["FieldingScore"], weights=g["play_weight"]),
                "first_year": int(g["yearID"].min() if len(g) else 0),
                "last_year": int(g["yearID"].max() if len(g) else 0),
            })
        ).reset_index(drop=True)
    )
    # bring back names
    if "playerName" in ranked.columns:
        names = ranked[["playerID","playerName"]].drop_duplicates()
        career = career.merge(names, on="playerID", how="left")
    career = career.sort_values("career_OverallScore", ascending=False)
    return ranked, career



def determine_primary_team_per_year(bat, fld):
    bat_team = (
        bat.groupby(["playerID", "yearID", "teamID"], as_index=False)["G"]
        .sum()
        .rename(columns={"G": "batG"})
    )

    fld_team = (
        fld.groupby(["playerID", "yearID", "teamID"], as_index=False)["G"]
        .sum()
        .rename(columns={"G": "fldG"})
    )

    # Merge
    merged = pd.merge(bat_team, fld_team, how="outer",
                      on=["playerID", "yearID", "teamID"]).fillna(0)
    merged["totalG"] = merged["batG"] + merged["fldG"]

    primary_team = (
        merged.sort_values(["playerID", "yearID", "totalG"], ascending=False)
        .groupby(["playerID", "yearID"], as_index=False)
        .first()
        [["playerID", "yearID", "teamID"]]
    )

    fallback_team = (
        merged.groupby(["playerID", "yearID"], as_index=False)
        .first()
        [["playerID", "yearID", "teamID"]]
    )

    primary_idx = set(zip(primary_team["playerID"], primary_team["yearID"]))
    fallback_idx = set(zip(fallback_team["playerID"], fallback_team["yearID"]))

    missing_idx = fallback_idx - primary_idx
    missing_idx_list = list(missing_idx)

    if missing_idx_list:
        # Add missing rows using fallback teams
        extra_rows = (
            fallback_team
            .set_index(["playerID", "yearID"])
            .loc[missing_idx_list]
            .reset_index()
        )

        primary_team = pd.concat([primary_team, extra_rows], ignore_index=True)

    return primary_team

def build_team_raw_totals(bat: pd.DataFrame, pit: pd.DataFrame, fld: pd.DataFrame) -> pd.DataFrame:
    bat_team = bat.groupby(["teamID","yearID"]).agg({
        "G":"sum","AB":"sum","R":"sum","H":"sum","2B":"sum","3B":"sum","HR":"sum",
        "RBI":"sum","SB":"sum","CS":"sum","BB":"sum","SO":"sum","HBP":"sum","SH":"sum","SF":"sum","GIDP":"sum"
    }).reset_index().add_prefix("bat_")
    bat_team = bat_team.rename(columns={"bat_teamID":"teamID","bat_yearID":"yearID"})

    pit_team = pit.groupby(["teamID","yearID"]).agg({
        "W":"sum","L":"sum","G":"sum","GS":"sum","CG":"sum","SHO":"sum","SV":"sum",
        "IPouts":"sum","H":"sum","ER":"sum","HR":"sum","BB":"sum","SO":"sum",
        "WP":"sum","HBP":"sum","BK":"sum","BFP":"sum","GF":"sum","R":"sum"
    }).reset_index().add_prefix("pit_")
    pit_team = pit_team.rename(columns={"pit_teamID":"teamID","pit_yearID":"yearID"})

    fld_team = fld.groupby(["teamID","yearID"]).agg({
        "G":"sum","PO":"sum","A":"sum","E":"sum","DP":"sum","PB":"sum","WP":"sum","SB":"sum","CS":"sum"
    }).reset_index().add_prefix("fld_")
    fld_team = fld_team.rename(columns={"fld_teamID":"teamID","fld_yearID":"yearID"})

    ts = bat_team.merge(pit_team, on=["teamID","yearID"], how="outer").merge(fld_team, on=["teamID","yearID"], how="outer").fillna(0)
    return ts

def build_team_weighted_scores(
    ranked: pd.DataFrame,
    bat: pd.DataFrame,
    fld: pd.DataFrame,
    app: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    primary_team = determine_primary_team_per_year(bat, fld)

    ranked_team = ranked.merge(primary_team, on=["playerID","yearID"], how="left")

    usage_df = None

    if app is not None:
        app = app.copy()

        g_col = None
        for cand in ["G_all", "G_batting", "G"]:
            if cand in app.columns:
                g_col = cand
                break

        if g_col is not None:
            app_team = (
                app.groupby(["playerID","yearID","teamID"], as_index=False)[g_col]
                   .sum()
                   .rename(columns={g_col: "G_player_team"})
            )

            # Merge into ranked_team (on player, year, team)
            ranked_team = ranked_team.merge(
                app_team,
                on=["playerID","yearID","teamID"],
                how="left",
            )

            ranked_team["G_player_team"] = ranked_team["G_player_team"].fillna(0.0)
            team_tot_games = (
                ranked_team.groupby(["teamID","yearID"])["G_player_team"]
                           .transform("sum")
            )

            ranked_team["usage_weight"] = np.where(
                team_tot_games > 0,
                ranked_team["G_player_team"] / team_tot_games,
                1.0,
            )

            usage_df = ranked_team[[
                "playerID","teamID","yearID","usage_weight"
            ]].dropna(subset=["teamID"]).copy()
        else:
            print("[WARN] Appearances.csv has no recognizable games column; using equal player weights.")
            ranked_team["usage_weight"] = 1.0
    else:
        ranked_team["usage_weight"] = 1.0

    def agg_team(g: pd.DataFrame) -> pd.Series:
        w = g["usage_weight"].to_numpy(dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()

        return pd.Series({
            "team_BattingScore":  float(np.average(g["BattingScore"],  weights=w)),
            "team_PitchingScore": float(np.average(g["PitchingScore"], weights=w)),
            "team_FieldingScore": float(np.average(g["FieldingScore"], weights=w)),
            "team_OverallScore":  float(np.average(g["OverallScore"], weights=w)),
            "team_play_weight":   float(len(g)),
        })

    team_weighted = (
        ranked_team.dropna(subset=["teamID"])
                   .groupby(["teamID","yearID"])
                   .apply(agg_team)
                   .reset_index()
    )

    return team_weighted, usage_df


if TORCH_OK:

    class BaseballBettingEnv:
        """
        Episodic RL environment.

        - One episode = one pass through all games in a random order (a "season").
        - State  = feature vector for the current game (home+away+bias).
        - Action = 0 (bet Away) or 1 (bet Home).
        - Reward = +1 if the bet matches the actual winner, else -1.
        - done   = True only after we've gone through all games in the episode.
        """

        def __init__(self, X, y):
            self.X = X.astype(np.float32)
            self.y = y.astype(np.int64)
            self.n = len(self.y)
            self.rng = np.random.default_rng(SEED)
            self.order = np.arange(self.n, dtype=int)
            self.idx = 0

        def _shuffle_season(self):
            """Shuffle the order of games for a new episode."""
            self.order = self.rng.permutation(self.n)
            self.idx = 0

        def reset(self):
            """
            Start a new episode (new shuffled season) and return the first state.
            """
            self._shuffle_season()
            # First game in the shuffled order
            game_idx = self.order[self.idx]
            return self.X[game_idx]

        def step(self, action: int):
            """
            Advance one game in the current episode.

            Returns:
                next_state: features for the next game (or a dummy state if done)
                reward:     +1 if prediction correct, -1 otherwise
                done:       True when we've reached the end of the season
            """
            # Current game index in the shuffled order
            game_idx = self.order[self.idx]
            true = self.y[game_idx]

            # Reward: did we pick the correct winner?
            reward = 1.0 if action == true else -1.0

            # Move to the next game
            self.idx += 1
            done = (self.idx >= self.n)

            if done:
                # Episode finished: we return a dummy state (it will be masked out by 'done')
                next_state = np.zeros_like(self.X[0])
            else:
                # Next game in the shuffled order
                next_game_idx = self.order[self.idx]
                next_state = self.X[next_game_idx]

            return next_state, reward, done

    class PrioritizedReplay:
        def __init__(self, capacity=50000, alpha=0.6):
            self.capacity = capacity
            self.alpha = alpha
            self.pos = 0
            self.data = []
            self.priorities = []

        def __len__(self):
            return len(self.data)

        def push(self, transition, priority=1.0):
            if len(self.data) < self.capacity:
                self.data.append(transition)
                self.priorities.append(priority)
            else:
                self.data[self.pos] = transition
                self.priorities[self.pos] = priority
                self.pos = (self.pos + 1) % self.capacity

        def sample(self, batch_size, beta=0.4):
            if len(self.data) == 0:
                raise ValueError("PER empty")
            prios = torch.tensor(self.priorities, dtype=torch.float32)
            probs = prios.pow(self.alpha)
            probs /= probs.sum()
            idx = torch.multinomial(probs, batch_size, replacement=True)
            samples = [self.data[i] for i in idx]
            weights = (len(self.data) * probs[idx]).pow(-beta)
            weights /= weights.max()
            return samples, idx, weights

        def update_priorities(self, idx, prios):
            for i, p in zip(idx.tolist(), prios.tolist()):
                self.priorities[i] = float(p)

    class DuelingDQN(nn.Module):
        def __init__(self, input_dim, hidden=(256,128), n_actions=2):
            super().__init__()
            self.feature = nn.Sequential(
                nn.Linear(input_dim, hidden[0]),
                nn.ReLU(),
                nn.Linear(hidden[0], hidden[1]),
                nn.ReLU(),
            )
            self.value = nn.Sequential(
                nn.Linear(hidden[1], 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            self.adv   = nn.Sequential(
                nn.Linear(hidden[1], 128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
            )

        def forward(self, x):
            h = self.feature(x)
            v = self.value(h)
            a = self.adv(h)
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q

def build_game_matrix(games_df: pd.DataFrame, team_df: pd.DataFrame):
    team_cols = [c for c in team_df.columns if c not in ("teamID","yearID")]

    g = games_df.copy()
    g["yearID"] = pd.to_datetime(g["date"]).dt.year
    g["home_win"] = (g["homeScore"] > g["awayScore"]).astype(int)

    # Merge home and away team stats
    home = g.merge(team_df, left_on=["homeTeam", "yearID"],
                   right_on=["teamID", "yearID"], how="left")
    for c in team_cols:
        home = home.rename(columns={c: f"home_{c}"})

    away = home.merge(team_df, left_on=["awayTeam", "yearID"],
                      right_on=["teamID", "yearID"], how="left")
    for c in team_cols:
        away = away.rename(columns={c: f"away_{c}"})

    away = away.fillna(0).infer_objects(copy=False)

    # Build diff features (home - away)
    diff_feats = []
    for c in team_cols:
        h = f"home_{c}"
        a = f"away_{c}"
        d = f"diff_{c}"
        away[d] = away[h] - away[a]
        diff_feats.append(d)

    # Final feature list: home, away, diff
    home_feats = [f"home_{c}" for c in team_cols]
    away_feats = [f"away_{c}" for c in team_cols]
    feat_cols = home_feats + away_feats + diff_feats

    X = away[feat_cols].values.astype(np.float32)

    # bias term
    X = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])

    y = away["home_win"].values.astype(np.int64)
    years = away["yearID"].values.astype(np.int64)   # <--- NEW

    return X, y, years, feat_cols, team_cols

def build_enhanced_game_matrix(games_df, team_df):
    X_base, y, years, feat_cols, team_cols = build_game_matrix(games_df, team_df)

    # Additional per-game features
    extra_cols = [
        "home_roll_runs_5","home_roll_runs_10","home_roll_winpct_10",
        "away_roll_runs_5","away_roll_runs_10","away_roll_winpct_10",
        "temp","wind","humidity","windspeed"
    ]
    used = [c for c in extra_cols if c in games_df.columns]

    if used:
        extra = games_df[used].fillna(0).to_numpy(dtype=np.float32)
        X = np.hstack([X_base, extra])
        feat_cols = feat_cols + used
    else:
        X = X_base

    return X, y, years, feat_cols, team_cols

def build_forecast_matrix(schedule_df: pd.DataFrame, team_df: pd.DataFrame, ref_year: int):

    # Use team stats from ref_year as proxy for future strength
    team_year = team_df[team_df["yearID"] == ref_year].copy()
    if team_year.empty:
        raise ValueError(f"No team features found for reference year {ref_year}.")

    team_cols = [c for c in team_year.columns if c not in ("teamID", "yearID")]

    g = schedule_df.copy()
    g["date"] = pd.to_datetime(g["date"])
    g["yearID"] = g["date"].dt.year   # this will be forecast_year
    # Dummy label (we don't know the result); just placeholder
    g["home_win"] = 0

    home = g.merge(team_year, left_on="homeTeam", right_on="teamID", how="left")
    for c in team_cols:
        home = home.rename(columns={c: f"home_{c}"})

    away = home.merge(team_year, left_on="awayTeam", right_on="teamID", how="left")
    for c in team_cols:
        away = away.rename(columns={c: f"away_{c}"})

    away = away.fillna(0).infer_objects(copy=False)

    diff_feats = []
    for c in team_cols:
        h = f"home_{c}"
        a = f"away_{c}"
        d = f"diff_{c}"
        away[d] = away[h] - away[a]
        diff_feats.append(d)

    home_feats = [f"home_{c}" for c in team_cols]
    away_feats = [f"away_{c}" for c in team_cols]
    feat_cols = home_feats + away_feats + diff_feats

    X = away[feat_cols].values.astype(np.float32)
    X = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])  # bias term

    years = away["yearID"].values.astype(np.int64)
    return X, years, feat_cols

def train_logistic_ridge(X, y, seed=SEED, C=1.0):

    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y,
        test_size=0.2,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    model = LogisticRegression(
        penalty='l2',
        C=C,
        max_iter=1000,
        solver='lbfgs',
        n_jobs=-1
    )

    model.fit(Xtr, ytr)
    acc = model.score(Xte, yte)

    return model, scaler, acc

def train_gb(X, y, seed=SEED):

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y,
        test_size=0.2,
        random_state=seed,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    if USE_XGB:
        gb_model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",          # hist is the fast algo
            device="cuda",               # <-- THIS turns on GPU
            predictor="auto",            # let xgboost pick GPU/CPU predictor
            random_state=seed,
            eval_metric="logloss",
        )

    else:
        gb_model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=1,
            random_state=seed,
        )

    gb_model.fit(Xtr, ytr)
    acc = gb_model.score(Xte, yte)

    return gb_model, scaler, acc

def train_lr_and_gb_time_split(
    X,
    y,
    years,
    train_start=2014,
    train_end=2023,
    test_year=2024,
    seed=SEED,
):
    """
    Train Logistic Regression and Gradient Boosting on a time-based split.

    - Train on games with train_start <= year <= train_end
    - Test  on games with year == test_year

    Returns:
        {
          "lr":   {"model": ..., "scaler": ..., "acc": ..., "baseline_acc": ...},
          "gb":   {"model": ..., "scaler": ..., "acc": ..., "baseline_acc": ...},
          "meta": {"home_win_rate_train": ..., "home_win_rate_test": ...}
        }
    """
    years = np.asarray(years, dtype=int)
    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)

    # Masks
    train_mask = (years >= train_start) & (years <= train_end)
    test_mask  = (years == test_year)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Train/test split produced empty sets. Check year ranges.")

    # Baseline: majority class in TRAIN
    home_win_rate_train = y_train.mean()
    home_win_rate_test  = y_test.mean()
    baseline_acc_train  = max(home_win_rate_train, 1 - home_win_rate_train)
    baseline_acc_test   = max(home_win_rate_test, 1 - home_win_rate_test)

    print(f"[SPLIT] Train years: {train_start}-{train_end}, "
          f"Test year: {test_year}")
    print(f"[SPLIT] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"[BASELINE] Train home win rate: {home_win_rate_train:.3f}, "
          f"baseline acc: {baseline_acc_train*100:.2f}%")
    print(f"[BASELINE] Test home win rate:  {home_win_rate_test:.3f}, "
          f"baseline acc: {baseline_acc_test*100:.2f}%")

    # Logistic Regression (PyTorch on GPU if available)
    scaler_lr = StandardScaler()
    Xtr_lr = scaler_lr.fit_transform(X_train).astype(np.float32)
    Xte_lr = scaler_lr.transform(X_test).astype(np.float32)

    if TORCH_OK and DEVICE is not None and DEVICE.type == "cuda":
        # Move data to GPU
        Xtr_t = torch.from_numpy(Xtr_lr).to(DEVICE)
        ytr_t = torch.from_numpy(y_train.astype(np.float32)).to(DEVICE)
        Xte_t = torch.from_numpy(Xte_lr).to(DEVICE)
        yte_t = torch.from_numpy(y_test.astype(np.float32)).to(DEVICE)

        # Simple logistic regression model: Linear + sigmoid
        lr_model = nn.Linear(Xtr_lr.shape[1], 1).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(lr_model.parameters(), lr=1e-3)

        n_epochs = 200
        batch_size = 512

        for epoch in range(n_epochs):
            # mini-batch SGD
            perm = torch.randperm(Xtr_t.size(0), device=DEVICE)
            for i in range(0, Xtr_t.size(0), batch_size):
                idx = perm[i:i + batch_size]
                xb = Xtr_t[idx]
                yb = ytr_t[idx].unsqueeze(1)

                optimizer.zero_grad()
                logits = lr_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            logits_test = lr_model(Xte_t).squeeze(1)
            probs_test = torch.sigmoid(logits_test)
            preds_test = (probs_test > 0.5).long()
            lr_acc = (preds_test.cpu().numpy() == y_test).mean()

        print(f"[LOGISTIC CUDA] Time-split test accuracy: {lr_acc*100:.2f}%")

    else:

        lr_model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
            random_state=seed,
        )
        lr_model.fit(Xtr_lr, y_train)
        lr_acc = lr_model.score(Xte_lr, y_test)
        print(f"[LOGISTIC CPU] Time-split test accuracy: {lr_acc*100:.2f}%")


    scaler_gb = StandardScaler()
    Xtr_gb = scaler_gb.fit_transform(X_train)
    Xte_gb = scaler_gb.transform(X_test)

    if USE_XGB:
        gb_model = xgb.XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",          # hist is the fast algo
            device="cuda",               # <-- THIS turns on GPU
            predictor="auto",            # let xgboost pick GPU/CPU predictor
            random_state=seed,
            eval_metric="logloss",
        )

    else:
        gb_model = GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            random_state=seed,
        )

    gb_model.fit(Xtr_gb, y_train)
    gb_acc = gb_model.score(Xte_gb, y_test)
    print(f"[GRAD_BOOST/XGB] Time-split test accuracy: {gb_acc*100:.2f}%")


    return {
        "lr": {
            "model": lr_model,
            "scaler": scaler_lr,
            "acc": lr_acc,
            "baseline_acc": baseline_acc_test,
        },
        "gb": {
            "model": gb_model,
            "scaler": scaler_gb,
            "acc": gb_acc,
            "baseline_acc": baseline_acc_test,
        },
        "meta": {
            "home_win_rate_train": home_win_rate_train,
            "home_win_rate_test": home_win_rate_test,
        },
    }

def predict_home_prob(model, scaler, home_vec, away_vec):

    home_vec = np.asarray(home_vec, dtype=np.float32)
    away_vec = np.asarray(away_vec, dtype=np.float32)

    # Rebuild diff features the same way build_game_matrix did
    diff_vec = home_vec - away_vec

    x = np.hstack([
        home_vec,
        away_vec,
        diff_vec,
        np.array([1.0], dtype=np.float32)  # bias term
    ]).reshape(1, -1)

    xs = scaler.transform(x)
    p = model.predict_proba(xs)[0][1]  # P(home win)

    return float(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing Lahman CSVs AND YYYYteamstats.csv files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2014,
        help="First season year to use from YYYYteamstats.csv",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Last season year to use from YYYYteamstats.csv",
    )
    parser.add_argument(
        "--forecast-year",
        type=int,
        default=None,
        help="Future season year to forecast (e.g. 2025). Requires YYYYschedule.csv in data-dir.",
    )
    parser.add_argument(
        "--forecast-schedule",
        type=str,
        default=None,
        help="Optional explicit path to a future schedule CSV; if not given, uses <data-dir>/<forecast-year>schedule.csv",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    people, bat, pit, fld = load_csvs(data_dir)

    bat_war = compute_basic_batting_war(bat)
    pit_war = compute_basic_pitching_war(pit)

    war = pd.merge(
        bat_war,
        pit_war,
        on=["playerID", "yearID"],
        how="outer",
    ).fillna(0.0)

    war["WAR_basic_total"] = war["Bat_WAR_basic"] + war["Pit_WAR_basic"]

    war = war.sort_values(["yearID", "WAR_basic_total"], ascending=[True, False])

    war.to_csv(out_dir / "player_WAR_basic.csv", index=False)
    print(f"[WAR] Wrote basic WAR-style estimates to {out_dir / 'player_WAR_basic.csv'}")

    app = None
    app_path = data_dir / "Appearances.csv"
    if app_path.exists():
        app = pd.read_csv(app_path)
        print(f"[INFO] Loaded Appearances.csv with {len(app)} rows")
    else:
        print("[WARN] Appearances.csv not found – using equal weights per player for team scores.")

    plyr = build_player_year(people, bat, pit, fld)
    ranked = build_subscores_per_year(plyr)
    ranked, career = build_career(ranked, pit)

    per_year_cols = [
        "yearID", "playerID", "playerName",
        "BattingScore", "PitchingScore", "FieldingScore",
        "OverallScore", "rank_in_year",
        "AB", "H", "HR", "RBI", "OPS", "ERA", "WHIP", "FPCT", "RF",
    ]
    for c in per_year_cols:
        if c not in ranked.columns:
            ranked[c] = np.nan

    ranked[per_year_cols]\
        .sort_values(["yearID", "rank_in_year"])\
        .to_csv(out_dir / "player_rankings_by_year.csv", index=False)

    # Save career rankings
    career_cols = [
        "playerID", "playerName", "first_year", "last_year",
        "career_BattingScore", "career_PitchingScore",
        "career_FieldingScore", "career_OverallScore",
    ]
    for c in career_cols:
        if c not in career.columns:
            career[c] = np.nan

    career[career_cols]\
        .sort_values("career_OverallScore", ascending=False)\
        .to_csv(out_dir / "player_rankings_career.csv", index=False)

    # Team totals + usage-weighted scores
    team_totals = build_team_raw_totals(bat, pit, fld)
    team_weighted, usage_df = build_team_weighted_scores(ranked, bat, fld, app)
    team_stats = (
        team_totals
        .merge(team_weighted, on=["teamID", "yearID"], how="left")
        .fillna(0.0)
    )

    numeric_cols = [c for c in team_stats.columns if c not in ("teamID", "yearID") and
                    np.issubdtype(team_stats[c].dtype, np.number)]

    if not numeric_cols:
        print("[CLUSTER] No numeric team columns found for clustering; skipping compression.")
    else:
        original_team_stats = team_stats.copy()

        X_raw = original_team_stats[numeric_cols].fillna(0).values.astype(float)

        scaler_k = StandardScaler()
        Xs = scaler_k.fit_transform(X_raw)

        n_clusters = min(N_CLUSTERS, max(1, Xs.shape[0]))

        if n_clusters <= 1:
            labels = np.zeros(Xs.shape[0], dtype=int)
            dists = np.zeros((Xs.shape[0], 1), dtype=float)
            kmeans = None
            print("[CLUSTER] Only one cluster possible; assigned cluster 0 to all rows.")
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
            kmeans.fit(Xs)
            labels = kmeans.predict(Xs)
            dists = kmeans.transform(Xs)  # distances to each centroid

        compressed = pd.DataFrame({
            "teamID": original_team_stats["teamID"].values,
            "yearID": original_team_stats["yearID"].values,
            "cluster_id": labels,
            "cluster_dist_min": np.min(dists, axis=1),
            "cluster_dist_mean": np.mean(dists, axis=1),
        })

        for ci in range(dists.shape[1]):
            compressed[f"cluster_dist_{ci}"] = dists[:, ci]

        for ci in range(dists.shape[1]):
            compressed[f"cluster_oh_{ci}"] = (labels == ci).astype(int)

        team_stats = compressed.copy()

        model_dir = out_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": scaler_k,
            "kmeans": kmeans,
            "numeric_cols": numeric_cols,
            "n_clusters": dists.shape[1]
        }, model_dir / "team_kmeans.pkl")
        print(f"[CLUSTER] Saved cluster artifacts to {model_dir / 'team_kmeans.pkl'} (k={dists.shape[1]})")

    team_stats.to_csv(out_dir / "team_features_with_weighted_scores.csv", index=False)
    print(f"[TEAM] Saved team_features_with_weighted_scores.csv with {len(team_stats)} rows")


    yearly_ts = load_yearly_teamstats(data_dir, args.start_year, args.end_year)
    yearly_gi = load_yearly_gameinfo(data_dir, args.start_year, args.end_year)

    games = build_games_with_enhancements(yearly_ts, yearly_gi)

    X, y, years, feat_cols, team_cols = build_enhanced_game_matrix(games, team_stats)

    print("X shape:", X.shape)
    print("y shape:", y.shape)


    unique_years = np.sort(np.unique(years))
    if len(unique_years) < 2:
        print("[WARN] Only one year of games; skipping model training.")
        print(f"Done. Outputs written to: {out_dir.resolve()}")
        return

    train_start = int(unique_years[0])
    test_year = int(unique_years[-1])
    train_end = test_year - 1

    print(f"[SPLIT] Training on {train_start}-{train_end}, testing on {test_year}.")

    results = train_lr_and_gb_time_split(
        X,
        y,
        years,
        train_start=train_start,
        train_end=train_end,
        test_year=test_year,
    )


    gb_model  = results["gb"]["model"]
    gb_scaler = results["gb"]["scaler"]
    baseline  = results["lr"]["baseline_acc"]
    lr_acc    = results["lr"]["acc"]
    gb_acc    = results["gb"]["acc"]

    print(f"[SUMMARY] Baseline test acc: {baseline*100:.2f}%")
    print(f"[SUMMARY] LOGISTIC test acc: {lr_acc*100:.2f}%")
    print(f"[SUMMARY] GRAD_BOOST test acc: {gb_acc*100:.2f}%")


    test_mask = (years == test_year)
    X_test = X[test_mask]
    games_test = games.loc[test_mask].copy()

    Xs_test = gb_scaler.transform(X_test)
    probs = gb_model.predict_proba(Xs_test)[:, 1]  # P(home win)

    games_test["pred_home_win_prob"] = probs
    games_test["pred_home_win_pct"] = probs * 100.0
    out_pred = out_dir / f"games_with_predicted_win_prob_{test_year}.csv"
    games_test.to_csv(out_pred, index=False)
    print(f"[OUT] Saved game predictions for {test_year} to {out_pred}")

    # Optional so as to Forecast a future season
    if args.forecast_year is not None:
        forecast_year = args.forecast_year

        ref_year = int(unique_years[-1])  # same as test_year

        if args.forecast_schedule is not None:
            sched_path = Path(args.forecast_schedule)
        else:
            sched_path = data_dir / f"{forecast_year}schedule.csv"

        if not sched_path.exists():
            print(f"[FORECAST] No schedule file found at {sched_path}, skipping forecast.")
        else:
            print(f"[FORECAST] Loading schedule from {sched_path}")
            schedule_df = pd.read_csv(sched_path)


            required_cols = {"date", "homeTeam", "awayTeam"}
            missing = required_cols - set(schedule_df.columns)
            if missing:
                raise ValueError(f"Schedule is missing required columns: {missing}")

            X_fore, years_fore, _ = build_forecast_matrix(schedule_df, team_stats, ref_year)


            n_train = gb_scaler.n_features_in_
            n_fore = X_fore.shape[1]
            if n_fore < n_train:

                pad = np.zeros((X_fore.shape[0], n_train - n_fore), dtype=X_fore.dtype)
                X_fore = np.hstack([X_fore, pad])
            elif n_fore > n_train:

                X_fore = X_fore[:, :n_train]


            Xs_fore = gb_scaler.transform(X_fore)
            probs_fore = gb_model.predict_proba(Xs_fore)[:, 1]

            schedule_df["pred_year"] = forecast_year
            schedule_df["pred_home_win_prob"] = probs_fore
            schedule_df["pred_home_win_pct"] = probs_fore * 100.0

            out_fore = out_dir / f"games_with_predicted_win_prob_{forecast_year}.csv"
            schedule_df.to_csv(out_fore, index=False)
            print(f"[FORECAST] Saved {forecast_year} forecasts to {out_fore}")

    print(f"Done. Outputs written to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
