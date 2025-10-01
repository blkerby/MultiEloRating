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
import torch

# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# x = torch.arange(-5, 5, 0.01)
# y = torch.exp(multi_elo_rating.exp_modified_gaussian_log_density(1)(x))
# plt.plot(x, y)
# plt.show()

parser = argparse.ArgumentParser(
    prog='model_analysis',
    description='Analysis using model scores')
parser.add_argument("goal", help="Racetime goal (e.g., Map Rando S4)")
parser.add_argument("data_dir", help="Directory to load data from")
parser.add_argument("--trueskill", action="store_true")
parser.add_argument("--dummy", action="store_true", help="Include dummy player")
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()

data_path = os.path.join(args.data_dir, f"{args.goal}.json")
race_list = json.load(open(data_path, "r"))

# Filter to recorded races:
race_list = [race for race in race_list if race["recorded"]]

# trueskill_env = trueskill.TrueSkill(backend='mpmath')
trueskill_env = trueskill.TrueSkill(backend=None, sigma=5.0, beta=2.5, tau=0.12, draw_probability=1e-3)
# trueskill_env = trueskill.TrueSkill(backend=None, sigma=3.0, beta=2.0, tau=0.12, draw_probability=1e-3)
# trueskill_env = trueskill.TrueSkill(backend=None, sigma=3.0, beta=2.0, tau=0.12, draw_probability=1e-3)
# trueskill_env = trueskill.TrueSkill(backend=None, sigma=3.7, beta=2.3, tau=0.18, draw_probability=1e-3)
# trueskill_env = trueskill.TrueSkill(backend=None, sigma=3.6, beta=3.0, tau=0.15, draw_probability=1e-5)

# trueskill_env = trueskill.TrueSkill(backend=None, mu=0.0, sigma=8.0, beta=5.5, tau=0.4, draw_probability=0.01)

user_scores = {}
user_scores_history = defaultdict(lambda: [])

dummy_player_weight = 1.0

# Optimized values for TrueSkill:
default_score = 25
dummy_sigma = 0.01
dummy_score = 21
rating_floor = -float("inf")

# Optimized values for Thurstonian model:
# default_score = 0.1
# dummy_score = -0.1
# rating_floor = 0.0

# rating_floor = 0.0
# default_score = 0.25
# dummy_score = 1.35

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
            score1 = score1.mu - 2 * score1.sigma
            # score1 = score1.mu
        for j in range(i + 1, len(entrants)):
            e2 = entrants[j]
            user2 = e2["user_id"]
            if user2 is None:
                continue
            score2 = user_scores.get(user2, default_rating)
            if args.trueskill:
                score2 = score2.mu - 2 * score2.sigma
                # score2 = score2.mu
            if e1["place"] == e2["place"]:
                continue
            if score1 == score2 or np.isnan(score1) or np.isnan(score2):
                cnt_incorrect += 0.5
            else:
                incorrect = (e1["place"] < e2["place"]) == (score1 < score2)
                if incorrect:
                    cnt_incorrect += 1
            cnt_pair += 1
    return cnt_incorrect, cnt_pair

def score_race(race_idx, entrants):
    if args.trueskill:
        default_rating = trueskill_env.create_rating(mu=default_score)
    else:
        default_rating = default_score

    user_id_list = []
    rating_list = []
    rank_list = []
    weight_list = []
    entrant_idx_list = []
    for i, e in enumerate(entrants):
        user_id = e["user_id"]
        if user_id is None:
            continue
        rating = user_scores.get(user_id, default_rating)
        rank = e["place"]
        user_id_list.append(user_id)
        rating_list.append(rating)
        rank_list.append(rank)
        weight_list.append(1)
        entrant_idx_list.append(i)

    if args.dummy:
        # Add a dummy player tied for last place, with a constant rating, to anchor the rating scale:
        if args.trueskill:
            rating_list.append(trueskill.Rating(mu=dummy_score, sigma=dummy_sigma))
        else:
            rating_list.append(dummy_score)
        rank_list.append(entrants[-1]["place"])
        weight_list.append(dummy_player_weight)

    if args.trueskill:
        wrapped_rating_list = [(x,) for x in rating_list]
        new_ratings = trueskill_env.rate(wrapped_rating_list, rank_list)
        new_ratings = [x[0] for x in new_ratings]
        for i in range(len(new_ratings)):
            new_ratings[i] = trueskill.Rating(mu=max(rating_floor, new_ratings[i].mu), sigma=new_ratings[i].sigma)
    else:                  
        ######## Constant learning rate, no dummy player:
    
        # ratings_grad = multi_elo_rating.pairwise_elo_gradient(
        #     rating_list,
        #     rank_list,
        #     use_average=False)
        # new_ratings = multi_elo_rating.get_updated_ratings_simple(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.07,
        # )

        # ratings_grad = multi_elo_rating.pairwise_elo_gradient(
        #     rating_list,
        #     rank_list,
        #     use_average=True)
        # new_ratings = multi_elo_rating.get_updated_ratings_simple(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.75,
        # )
        
        # ratings_grad = multi_elo_rating.plackett_luce_gradient_with_ranks(
        #     rating_list,
        #     rank_list)
        # new_ratings = multi_elo_rating.get_updated_ratings_simple(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.32,
        # )

        # ratings_grad = multi_elo_rating.get_thurstonian_rating_gradient(
        #     rating_list,
        #     rank_list,
        #     weight_list,
        #     log_density_fn=multi_elo_rating.gaussian_log_density,
        #     margin=5)
        # new_ratings = multi_elo_rating.get_updated_ratings_simple(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.26,
        # )


        ######## Constant learning rate, with dummy player:
    
        # ratings_grad = multi_elo_rating.pairwise_elo_gradient(
        #     rating_list,
        #     rank_list,
        #     use_average=False)
        # new_ratings = multi_elo_rating.get_updated_ratings_simple(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.07,
        # )

        # ratings_grad = multi_elo_rating.pairwise_elo_gradient(
        #     rating_list,
        #     rank_list,
        #     use_average=True)
        # new_ratings = multi_elo_rating.get_updated_ratings_simple(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.63,
        # )
        
        # ratings_grad = multi_elo_rating.plackett_luce_gradient_with_ranks(
        #     rating_list,
        #     rank_list)
        # new_ratings = multi_elo_rating.get_updated_ratings_simple(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.18,
        # )

        # ratings_grad = multi_elo_rating.get_thurstonian_rating_gradient(
        #     rating_list,
        #     rank_list,
        #     weight_list,
        #     log_density_fn=multi_elo_rating.gaussian_log_density,
        #     margin=5)
        # new_ratings = multi_elo_rating.get_updated_ratings_simple(
        #     rating_list,
        #     ratings_grad,
        #     learning_rate=0.11,
        # )


        ####### Variable learning rate, with rating floor:
        ratings_grad = multi_elo_rating.plackett_luce_gradient_with_ranks(
            rating_list,
            rank_list)
        new_ratings = multi_elo_rating.get_updated_ratings(
            rating_list,
            ratings_grad,
            rating_floor=rating_floor,
            knots=[0.0, 1.0, 2.0],
            learning_rates=[0.6, 0.13, 0.09],
        )

        # ratings_grad = multi_elo_rating.get_thurstonian_rating_gradient(
        #     rating_list,
        #     rank_list,
        #     weight_list,
        #     log_density_fn=multi_elo_rating.gaussian_log_density,
        #     margin=5)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     rating_list,
        #     ratings_grad,
        #     rating_floor=rating_floor,
        #     knots=[0.0, 1.0, 2.0],
        #     learning_rates=[0.65, 0.09, 0.07],
        # )

        # ratings_grad = multi_elo_rating.get_thurstonian_rating_gradient(
        #     rating_list,
        #     rank_list,
        #     weight_list,
        #     log_density_fn=multi_elo_rating.skew_gaussian_log_density(-1.0),
        #     margin=5)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     rating_list,
        #     ratings_grad,
        #     rating_floor=rating_floor,
        #     knots=[0.0, 1.0, 2.0],
        #     learning_rates=[0.7, 0.07, 0.07],
        # )

        # ratings_grad = multi_elo_rating.get_thurstonian_rating_gradient(
        #     rating_list,
        #     rank_list,
        #     weight_list,
        #     log_density_fn=multi_elo_rating.exp_modified_gaussian_log_density(1.0),
        #     margin=10)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     rating_list,
        #     ratings_grad,
        #     rating_floor=0,
        #     learning_rate_base=0.95,
        #     learning_rate_decay=0.6,
        #     negative_scaling_base=0.0,
        #     negative_scaling_decay=0.6,
        # )
        # new_ratings = multi_elo_rating.get_updated_ratings2(
        #     rating_list,
        #     ratings_grad,
        #     rating_floor=rating_floor,
        #     knots=[0.0, 1.0, 2.0, 3.0],
        #     learning_rates=[0.95, 0.5, 0.35, 0.1],
        # )

        # ratings_grad = multi_elo_rating.get_thurstonian_rating_gradient(
        #     rating_list,
        #     rank_list,
        #     weight_list,
        #     log_density_fn=multi_elo_rating.hyperbolic_exp_density,
        #     margin=10)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     rating_list,
        #     ratings_grad,
        #     rating_floor=0.0,
        #     learning_rate_base=0.95,
        #     learning_rate_decay=0.4,
        #     negative_scaling_base=0.0,
        #     negative_scaling_decay=0.45,
        # )

        # ratings_grad = multi_elo_rating.get_thurstonian_rating_gradient(
        #     rating_list,
        #     rank_list,
        #     weight_list,
        #     log_density_fn=multi_elo_rating.hyperbolic_secant_density,
        #     margin=10)
        # new_ratings = multi_elo_rating.get_updated_ratings(
        #     rating_list,
        #     ratings_grad,
        #     rating_floor=0.0,
        #     learning_rate_base=1.5,
        #     learning_rate_decay=0.8,
        #     negative_scaling_base=0.0,
        #     negative_scaling_decay=0.45,
        # )

    # if args.trueskill:
    #     for i in range(len(new_ratings)):
    #         adjusted_mu = new_ratings[i].mu - average_score * average_weight
    #         new_ratings[i] = trueskill.Rating(mu=adjusted_mu, sigma=new_ratings[i].sigma)
    # else:
    #     for i in range(len(new_ratings)):
    #         adjusted_mu = new_ratings[i] - average_score * average_weight
    #         new_ratings[i] = adjusted_mu
    for i in range(len(user_id_list)):
        user_id = user_id_list[i]
        old_rating = rating_list[i]
        new_rating = new_ratings[i]
        entrant = entrants[entrant_idx_list[i]]
        # if entrant["place"] == entrants[-1]["place"] and new_rating.mu > old_rating.mu:
        #     # print("last up: ", old_rating, new_rating, rank_list, rating_list, new_ratings)
        #     print(len(list(e for e in entrants if e["place"] == entrants[-1]["place"])))
        # if entrant["place"] == entrants[0]["place"] and new_rating < old_rating:
        #     print("first down")
        # if entrant["place"] == entrants[-1]["place"] and new_rating > old_rating:
        #     print(len(list(e for e in entrants if e["place"] == entrants[-1]["place"])))
        # if entrant["place"] == entrants[-1]["place"] and entrant["score_change"] is not None and entrant["score_change"] > 0:
        #     print(len(list(e for e in entrants if e["place"] == entrants[-1]["place"])))
        if args.baseline:
            new_rating = entrant["score"]
            if new_rating is not None and entrant["score_change"] is not None:
                new_rating += entrant["score_change"]
            else:
                continue
        user_scores[user_id] = new_rating
        user_scores_history[user_id].append((race_idx, new_rating))
        
    # print(cnt_above, cnt_below, len(new_ratings), average_score, default_score)

eval_pairs = []
for race_idx, race in enumerate(race_list):
    # winner_change = race["entrants"][0]["score_change"]
    # winner_change = race["entrants"][-1]["score_change"]
    # for e in race["entrants"]:
    #     if e["place"] == race["entrants"][0]["place"] and e["score_change"] < 0:
    #         print("first:", e["score_change"])
    #     if e["place"] == race["entrants"][-1]["place"] and e["score_change"] > 0:
    #         print("last:", e["score_change"])

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
# print(histories[0])

user_cnt5 = 0
user_win5 = 0
if args.trueskill:
    initial_score = default_score - 2 * trueskill_env.sigma
else:
    initial_score = default_score
for user_id, h in histories:
    if len(h) >= 5:
        user_cnt5 += 1
        race_idx, score = h[4]
        if args.trueskill:
            # score = score.mu
            score = score.mu - 2 * score.sigma
        if score > initial_score:
            user_win5 += 1
print("user_cnt5 = {}, user_win5={}, user_rate5={}".format(user_cnt5, user_win5, user_win5 / user_cnt5))

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
            # score_list.append(score.mu * 100)
            score_list.append((score.mu - 2 * score.sigma) * 100)
            # score_list.append(score.sigma)
        else:
            score_list.append(score)

print("kendall_tau: {}".format(kt))        
df = pd.DataFrame({
    "User ID": user_id_list,
    "Race": time_list,
    "Rating": score_list})


sns.set_theme()
sns.relplot(df, kind="line", x="Race", y="Rating", hue="User ID")
plt.show()
