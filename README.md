This is the assumed details on how to run programs due to minimal recording of how it was run at current development time
How to run weights.py:
weights.py is the backend program that is responsible for predicting baseball games using GPU and Pytorch with cuda.
1) Python 3.9 or newer needed to run program
2) pip install numpy pandas scikit-learn joblib
3) pip install xgboost torch
4) Basic  run -> python weights.py --data-dir data --out-dir output (Not the full program usage that was shot in video)
   Semi-Basic run -> python weights.py --data-dir data --out-dir output --start-year 2014 --end-year 2024
   (Not the full program usage that was shot in video)
   Full run -> python weights.py --data-dir data --out-dir output --start-year 2014 --end-year 2024 --forecast-year 2025
   (What was used in video and recomended)

create.py is used to create what will be the frontend so please run this before the actual frontend page
1) Python 3.9 or newer needed to run program
2) pip install pandas
3) Ensure that data folder has 2024allplayers.csv and People.csv and out folder has player_WAR_basic.csv
4) Run it with python create.py
5) Output should be output_players_war.csv to be used on frontend program

app.py is the fronted that will be the website to be used with all the data that was created
1) Python 3.9 or newer needed to run program
2) pip install flask pandas
3) Ensure that app.py is located on same folder as output_players_war.csv and that out folder has games_with_
predicted_win_prob_2025.csv and templates folder has index.html
4) Run with python app.py
5) Use any web browser and enter http://127.0.0.1:5000/
6) Website should show and terminal should show activity 
