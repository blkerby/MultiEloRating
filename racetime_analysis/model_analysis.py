import argparse
import logging
import os
import json
import numpy as np
import multi_elo_rating
import math
from collections import defaultdict
import datetime
import trueskill
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(
    prog='model_analysis',
    description='Analysis using model scores')
parser.add_argument("goal", help="Racetime goal (e.g., Map Rando S4)")
parser.add_argument("data_dir", help="Directory to load data from")
parser.add_argument("--trueskill", action="store_true")
args = parser.parse_args()

data_path = os.path.join(args.data_dir, f"{args.goal}.json")
race_list = json.load(open(data_path, "r"))

# Filter to recorded races:
race_list = [race for race in race_list if race["recorded"]]

# trueskill_env = trueskill.TrueSkill(backend='mpmath')
# trueskill_env = trueskill.TrueSkill(backend=None, mu=0.0, sigma=3.0, beta=3.5, tau=0.8, draw_probability=1e-3)
trueskill_env = trueskill.TrueSkill(backend=None, mu=0.0, sigma=8.0, beta=5.5, tau=0.4, draw_probability=0.01)

user_scores = {}
user_scores_history = defaultdict(lambda: [])

average_lr = 0.002
average_weight = 0.02
average_score = 0.0

default_lr = 0.1
default_quantile = 0.09
default_score = 0.0

# Evaluate using kendall-tau correlation:
def eval_race(entrants):
    if args.trueskill:
        default_rating = trueskill_env.create_rating(mu=default_score)
    else:
        default_rating = default_score

    cnt_pair = 0
    cnt_incorrect = 0
    for i in range(len(entrants)):
        e1 = entrants[i]
        user1 = e1["user_id"]
        if user1 is None:
            continue
        score1 = user_scores.get(user1, default_rating)
        if args.trueskill:
            score1 = score1.mu
        for j in range(i + 1, len(entrants)):
            e2 = entrants[j]
            user2 = e2["user_id"]
            if user2 is None:
                continue
            score2 = user_scores.get(user2, default_rating)
            if args.trueskill:
                score2 = score2.mu
            if e1["place"] == e2["place"]:
                continue
            if score1 == score2:
                cnt_incorrect += 0.5
            else:
                incorrect = (e1["place"] < e2["place"]) == (score1 < score2)
                if incorrect:
                    cnt_incorrect += 1
            cnt_pair += 1
    return cnt_incorrect, cnt_pair

def score_race(race_idx, entrants):
    # TODO: move these out of globals
    global average_score
    global default_score
    
    if args.trueskill:
        default_rating = trueskill_env.create_rating(mu=default_score)
    else:
        default_rating = default_score

    user_id_list = []
    rating_list = []
    rank_list = []
    for e in entrants:
        user_id = e["user_id"]
        if user_id is None:
            continue
        rating = user_scores.get(user_id, default_rating)
        rank = e["place"]
        user_id_list.append(user_id)
        rating_list.append(rating)
        rank_list.append(rank)

    if args.trueskill:
        rating_list = [(x,) for x in rating_list]
        new_ratings = trueskill_env.rate(rating_list, rank_list)
        new_ratings = [x[0] for x in new_ratings]
        for i in range(len(new_ratings)):
            adjusted_mu = new_ratings[i].mu - average_score * average_weight
            new_ratings[i] = trueskill.Rating(mu=adjusted_mu, sigma=new_ratings[i].sigma)
    else:                  
        # ratings_grad = multi_elo_rating.plackett_luce_gradient(
        #     rating_list,
        #     inverted=False)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.65,
        #     rating_floor=-0.1,
        #     negative_scaling=0.7,
        #     negative_scaling_cap=2.5,
        #     K_rate=1.5,
        # )

        ratings_grad = multi_elo_rating.plackett_luce_gradient(
            rating_list,
            inverted=True)
        new_ratings = multi_elo_rating.get_updated_ratings(
            rating_list,
            ratings_grad,
            learning_rate=0.45,
            rating_floor=-0.1,
            negative_scaling=0.75,
            negative_scaling_cap=2.5,
            K_rate=1.7,
        )

        # ratings_grad = multi_elo_rating.pairwise_elo_gradient(
        #     rating_list,
        #     rank_list,
        #     use_average=False)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.18,
        #     rating_floor=-0.1,
        #     negative_scaling=0.7,
        #     negative_scaling_cap=3.5,
        #     K_rate=1.85,
        # )

        # ratings_grad = multi_elo_rating.pairwise_elo_gradient(
        #     rating_list,
        #     rank_list,
        #     use_average=True)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=1.5,
        #     rating_floor=-0.1,
        #     negative_scaling=0.7,
        #     negative_scaling_cap=2.5,
        #     K_rate=1.5,
        # )

        # ratings_grad = multi_elo_rating.get_rating_gradient(
        #     rating_list,
        #     rank_list,
        #     log_density_fn=multi_elo_rating.gaussian_log_density,
        #     margin=5)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.85,
        #     rating_floor=-0.1,
        #     negative_scaling=0.7,
        #     negative_scaling_cap=3.5,
        #     K_rate=2.0,
        # )

        # ratings_grad = multi_elo_rating.get_rating_gradient(
        #     rating_list,
        #     rank_list,
        #     log_density_fn=multi_elo_rating.hyperbolic_exp_density,
        #     margin=10)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     ratings_grad,
        #     learning_rate=1.3,
        #     rating_floor=-0.2,
        #     negative_scaling=0.75,
        #     negative_scaling_cap=4.0,
        #     K_rate=1.3,
        # )

    for i in range(len(user_id_list)):
        user_id = user_id_list[i]
        new_rating = new_ratings[i]
        user_scores[user_id] = new_rating
        user_scores_history[user_id].append((race_idx, new_rating))
        
    if args.trueskill:
        race_average_score = sum(r.mu for r in new_ratings) / len(new_ratings)
    else:
        race_average_score = sum(new_ratings) / len(new_ratings)
    average_score = (1 - average_lr) * average_score + average_lr * race_average_score
    
    if args.trueskill:
        cnt_above = sum(1 for r in new_ratings if r.mu > default_score)
    else:
        cnt_above = sum(1 for r in new_ratings if r > default_score)        
    cnt_below = len(new_ratings) - cnt_above
    default_score += default_lr * (cnt_above * default_quantile - cnt_below * (1 - default_quantile))
    
    # print(cnt_above, cnt_below, len(new_ratings), average_score, default_score)

eval_pairs = []
for race_idx, race in enumerate(race_list):
    cnt_incorrect, cnt_pairs = eval_race(race["entrants"])
    if cnt_pairs > 0:
        eval_pairs.append((cnt_incorrect, cnt_pairs))
        score_race(race_idx, race["entrants"])
# print(eval_pairs)

kt = sum(x[0] for x in eval_pairs) / (sum(x[1] for x in eval_pairs))
if args.trueskill:
    max_score = max(s.mu for s in user_scores.values())
else:
    max_score = max(user_scores.values())

histories = list(user_scores_history.items())
histories.sort(key=lambda x: len(x[1]), reverse=True)
# print(list((x[0], len(x[1])) for x in histories))


user_id_list = []
time_list = []
score_list = []
for user_id, h in histories[:15]:
    for race_idx, score in h:
        user_id_list.append(user_id)
        # t = datetime.datetime.fromisoformat(race_list[race_idx]["ended_at"].replace('Z', '+00:00'))
        # time_list.append(t)
        time_list.append(race_idx)
        if args.trueskill:
            score_list.append(score.mu)
        else:
            score_list.append(score)

print("kendall_tau: {}".format(kt))        
df = pd.DataFrame({
    "User ID": user_id_list,
    "Race": time_list,
    "Simulated Rating": score_list})

sns.set_theme()
sns.relplot(df, kind="line", x="Race", y="Simulated Rating", hue="User ID")
# plt.show()
