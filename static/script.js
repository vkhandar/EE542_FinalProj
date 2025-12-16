document.addEventListener("DOMContentLoaded", () => {
    const homeSelect = document.getElementById("homeTeam");
    const awaySelect = document.getElementById("awayTeam");
    const result = document.getElementById("result");
    const btn = document.getElementById("predictBtn");

    btn.addEventListener("click", () => {
        const homeTeam = homeSelect.value;
        const awayTeam = awaySelect.value;

        if (!homeTeam || !awayTeam) {
            result.innerText = "Please choose both teams.";
            return;
        }

        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ homeTeam, awayTeam })
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                result.innerText = "No prediction available for this matchup.";
                return;
            }

            const pct = (data.probability * 100).toFixed(2);

            let homeList = data.home_players
                .map(p => `• ${p.nameFirst} ${p.nameLast}`)
                .join("<br>");

            let awayList = data.away_players
                .map(p => `• ${p.nameFirst} ${p.nameLast}`)
                .join("<br>");

            result.innerHTML = `
                <h2><strong>${homeTeam}</strong> Win Probability: <strong>${pct}%</strong></h2>
                
                <h3>Top 10 Players – ${homeTeam}</h3>
                <div>${homeList}</div>
        
                <h3>Top 10 Players – ${awayTeam}</h3>
                <div>${awayList}</div>
            `;
        })

        .catch(err => {
            result.innerText = "Error fetching prediction.";
        });
    });
});
