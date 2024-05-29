# Click run to run this program. The IP address will be displayed in the console which can be clicked to access the
# website. It takes a little bit of time for the popular games to be generated so give it some time to load the website

from flask import Flask, render_template, request
import requests
from Website_Recomms_Models.SVD_Website_Recomms import SVD_Recommend
from Website_Recomms_Models.Hybrid_Website_Recomm import Recommendations
from Website_Recomms_Models.CBF_Website_Recomms import get_user_recommendations

#initialise the Flask session
app = Flask(__name__)


# Makes a call to the Steam Spy API to get the top 100 games in the last 2 weeks. Takes 30 of those games and makes
# requests to the Steam Store API to get its information such as its image url.
# These games and their info is returned
def get_popular_games():
    top_games_ids = [entry for entry in requests.get("https://steamspy.com/api.php?request=top100in2weeks").json()]
    pop_recomms = []
    for gameid in top_games_ids[:30]:
        try:
            info = requests.get(f"https://store.steampowered.com/api/appdetails/?appids={gameid}").json()
            image = info[str(gameid)]['data']['header_image']
            name = info[str(gameid)]['data']['name']
            pop_recomms.append({"name": name, "image_url": image, "id": gameid})
        except:
            pass
    return pop_recomms


popular_games = get_popular_games()

# Defines the different methods and what they are meant to do. If its a post request, it makes the chosen model to
# recommend games to the user and it renders the template again with the generated recommendations.
@app.route('/', methods=['GET', 'POST'])
def index():
    steam_id = None
    recommendations = None
    if request.method == 'POST':
        steam_id = request.form['steamId']
        model = request.form['selectedModel']
        print(model)
        if model == "SVD":
            recommendations = SVD_Recommend(steam_id)
        elif model == "CBF":
            recommendations = get_user_recommendations(steam_id)
        else:
            recommendations = Recommendations(steam_id)
    return render_template('index.html', recommendations=recommendations, steam_id=steam_id, request_method=request.method, popular_games=popular_games)

if __name__ == '__main__':
    app.run(debug=True)
