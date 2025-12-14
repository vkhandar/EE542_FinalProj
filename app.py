from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)


df = pd.read_csv("out/games_with_predicted_win_prob_2025.csv")


teams = sorted(set(df["homeTeam"]).union(df["awayTeam"]))


players = pd.read_csv("output_players_war.csv")

top10_by_team = (
    players.sort_values(["teamID", "WAR_basic_total"], ascending=[True, False])
            .groupby("teamID")
            .head(10)
            .groupby("teamID")
            .apply(lambda g: g.to_dict(orient="records"))
            .to_dict()
)


@app.route("/")
def index():
    return render_template("index.html", teams=teams)

# @app.route("/predict", methods=["POST"])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    home = data.get("homeTeam")
    away = data.get("awayTeam")

    # Find prediction
    row = df[(df["homeTeam"] == home) & (df["awayTeam"] == away)]
    if row.empty:
        return jsonify({"error": "Matchup not found"}), 404

    prob = float(row.iloc[0]["pred_home_win_prob"])

    # Get top 10 players for each team
    home_players = top10_by_team.get(home, [])
    away_players = top10_by_team.get(away, [])

    return jsonify({
        "probability": prob,
        "home_players": home_players,
        "away_players": away_players
    })


if __name__ == "__main__":
    app.run(debug=True)
