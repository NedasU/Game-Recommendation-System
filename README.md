# Video Game Recommendation Systems Using User-Game Achievement Percentage and Additional Features

 The rise of video games and users has been rapidly growing in recent years.
 This can be clearly evident in the rise of the video game sectors revenue as it
 was only at $113.64 billion in 2017 and has risen to over $248.58 billion last
 year. Although this is an increase of over 118%, the rise has not been predicted
 to slow down any time soon as the predicted video game sectors revenue in the
 year 2027 has been predicted to reach $363.18 billion which would account to
 an increase of 46% from 2023 or 219% since 2017 Statista [2023]. This clearly
 shows the heavy demand for video games. Steam ste [b] is the most popular
 game distribution platform and has achieved a peak concurrent player count of
 33 million and additionally managing to generate $8.56 billion in revenue which
 shows its dominance as the total revenue for the PC (personal computer) market
 was around $37.6 billion in 2023 Clement [2024]Rutherford [2024]. Due to such
 a large demand for games, Steam’s main focus has been to deliver new games
 to their platform and ensure that their services work as desired. However, due
 to such a large variety of games, it can become difficult in picking a correct
 game. Recommendation systems can potentially connect users with games that
 may interest them. Recommendation Systems are an important factor in user
 experiences as it can potentially connect users to games that match their pref
erences which could improve the users experience and satisfaction. This is as it
 saves users a lot of time as they do not have to manually search through ran
dom games and instead, relevant games are recommended to the user. In recent
 years, Recommendation Systems have become very popular and have become
 a necessity for every successful business. However, in the video game sector,
 they have received relatively little exploration when compared to other forms of
 media, such as movies for Netflix. Steam only relatively recently released their
 “interactive recommender”– came out in 2020– which uses machine learning
 model that is trained using user playtime history, and their older RS which uses
 game tags in order to recommend similar items.
 
 Recommendation Systems designs can often be classified into two traditional
 categories: Collaborative Filtering (CF); and Content-Based Filtering (CBF).
 CF makes recommendations based on any users with similar tastes to the ac
tive user (the user who the recommendations are being made for). CBF works
 by using items features to make similarity comparisons based on those features
 and the users’ preferences and the most similar items are returned as the rec
ommendations. Hybrid recommendation systems are a combination of different
 techniques to utilize the best properties of each technique. Hybrid systems of
ten have more personalised recommendations but often have high computational
 complexity. In the case of video games– particularly the Steam platform- there
 are no way to rate games on a particular scale. As there is no explicit feedback
 available, it can make it relatively hard to develop an accurate model. For this
 reason, playtime is used as an indicator of enjoyment for a user. Although this
 is relatively accurate, it can lead to many outliers as not all games have the
 same playtime requirements for game completion. For example, MMORPG’s
(Massive multiplayer online role-playing game) can allow users to “AFK” (away
 from keyboard) in which the user is not required to be playing the game. When
 compared to an action game– such as Counter Strike– the game punishes
 users if the user goes AFK, especially in competitive environments. In order
 to minimise this problem and show better representation of user preferences,
 the use of completed achievement percentage (CAP) is introduced to be used
 alongside playtime in forming user ratings. This can potentially increase the
 performance of the models and the accuracy of the recommendations in which
 could improve user satisfaction. This paper explores the use of CAP as a factor
 in deciding user ratings in order to improve user preference representation in
 the data. Additionally, new, and unexplored item features are introduced such
 as the number of owners and review score of games to improve the efficiency
 and performance the CBF model. To provide a comprehensive evaluation, sev
eral models are employed. Alternating Least Square (ALS) and Singular Value
 Decomposition (SVD) models are used to evaluate the success of the CF models
 and the use of CAP. Additionally, a CBF model that uses cosine similarity as a
 similarity measure with additional features was created to test the effects of the
 new features. Lastly, a hybrid model that combines SVD and CBF was created
 to combine the strengths of both models and datasets to create an improved
 accuracy model. Finally, An interactive website was created to compare the
 recommendations generated by the best models using user Steam Ids

 *A Steam key is required to run these files. To generate a Steam Key, a valid steam account is required. This is a good guide for gathering your Steam API Key https://danbeyer.github.io/steamapi/page1.html. After getting your Key, replace the "key= ..." variable with your key as the value which will allow you to use the code.*
