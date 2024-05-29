# To run this program, either make a call from another file or make a call in this file.

import requests
import pandas as pd
import os
import json


# Forms the ratings in the range of 0-5 based on the rating score of the user when compared
# to the average playtime of the game
def form_rating(rating, avr_playtime):
    if rating > avr_playtime * 6:
        return 5
    elif rating > avr_playtime * 3:
        return 4
    elif rating > avr_playtime:
        return 3
    elif rating > avr_playtime * 0.5:
        return 2
    elif rating > avr_playtime * 0.2:
        return 1
    else:
        return 0


# Takes in the userid and a boolean value of whether to use achievements as a factor in deciding player ratings.
# Makes API calls to get the user games and their playtime. If achievmeent version is used, API calls are made for each
# valid game that the user owns. After forming the ratings, the user ratings are attached to the dataset and returned
# for the use of models in recommendations.
key = ...
# Must have a valid steam key from a valid steam account to access the steam dataset
def get_user_games(user_id, achivementver=True):
    with open("../Datasets/Average_Game_playtimeupdated.json") as avr_playtime_file:
        avr_game_playtimes = json.load(avr_playtime_file)

    user_file = pd.read_csv("../Datasets/User-Item-Interaction_2651.csv")
    valid_games = user_file['game_id'].unique()
    try:
        user_data = requests.get("https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key="
                             f"{key}&steamid={user_id}&include_played_free_games=true").json()

        user_games = {game['appid']: game['playtime_forever']
                      for game in user_data['response']['games'] if game['appid'] in valid_games}
    except:
        return None
    for game, playtime in user_games.items():
        if achivementver:
            achieved_count = 0
            info = requests.get("https://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/"
                                f"v1/?key={key}&steamid={user_id}&appid={game}").json()
            try:
                for achivdata in info['playerstats']['achievements']:
                    if achivdata['achieved'] == 1:
                        achieved_count += 1
                achievement_percentage = achieved_count / len(info['playerstats']['achievements'])
                user_games[game] += (user_games[game] * achievement_percentage)
            except:
                # Default average achievement achieved percentage is used to cover games that dont have achievements
                playtime += (playtime * 0.26978276146942665)

        user_games[game] = form_rating(playtime, avr_game_playtimes[str(game)])
    rows = []
    for game, rating in user_games.items():
        rows.append((10000, int(game), int(rating)))

    df = pd.DataFrame(rows, columns=['user_id', 'game_id', 'rating'])
    df = pd.concat([user_file, df], ignore_index=False)
    return df
