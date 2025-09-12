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

race_list = race_list

trueskill_env = trueskill.TrueSkill(backend='mpmath')
    
user_scores = {}
user_scores_history = defaultdict(lambda: [])
if args.trueskill:
    default_score = trueskill_env.create_rating()
else:
    default_score = 0.0

# Evaluate using kendall-tau correlation:
def eval_race(entrants):
    cnt_pair = 0
    cnt_incorrect = 0
    for i in range(len(entrants)):
        e1 = entrants[i]
        user1 = e1["user_id"]
        if user1 is None:
            continue
        score1 = user_scores.get(user1, default_score)
        if args.trueskill:
            score1 = score1.mu
        for j in range(i + 1, len(entrants)):
            e2 = entrants[j]
            user2 = e2["user_id"]
            if user2 is None:
                continue
            score2 = user_scores.get(user2, default_score)
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

def score_race(race_idx, entrants, learning_rate, score_floor, negative_scaling, negative_scaling_cap, K_rate):
    user_id_list = []
    rating_list = []
    rank_list = []
    for e in entrants:
        user_id = e["user_id"]
        if user_id is None:
            continue
        rating = user_scores.get(user_id, default_score)
        rank = e["place"]
        user_id_list.append(user_id)
        rating_list.append(rating)
        rank_list.append(rank)

    if args.trueskill:
        rating_list = [(x,) for x in rating_list]
        new_ratings = trueskill_env.rate(rating_list, rank_list)
        new_ratings = [x[0] for x in new_ratings]

        for i in range(len(user_id_list)):
            user_id = user_id_list[i]
            new_rating = new_ratings[i]
            user_scores[user_id] = new_rating
            user_scores_history[user_id].append((race_idx, new_rating))
    else:                    
        rating_grad = multi_elo_rating.get_rating_gradient(rating_list, rank_list)

        for i in range(len(user_id_list)):
            user_id = user_id_list[i]
            old_rating = rating_list[i]
            grad = rating_grad[i]
            if grad < 0:
                negative_scaling_strength = 1 - (min(old_rating, negative_scaling_cap) - score_floor) / (negative_scaling_cap - score_floor)
                grad *= 1 - negative_scaling * negative_scaling_strength
            rating_change = learning_rate * grad / (1 + K_rate * (old_rating - score_floor))
            new_rating = old_rating + rating_change
            if new_rating < score_floor:
                new_rating = score_floor
            user_scores[user_id] = new_rating
            user_scores_history[user_id].append((race_idx, new_rating))

recorded_only = True
learning_rate = 0.15
score_floor = -0.5
negative_scaling = 0.8
negative_scaling_cap = 3.5
K_rate = 0.6

eval_pairs = []
for race_idx, race in enumerate(race_list):
    if recorded_only and not race["recorded"]:
        continue
    cnt_incorrect, cnt_pairs = eval_race(race["entrants"])
    if cnt_pairs > 0:
        eval_pairs.append((cnt_incorrect, cnt_pairs))
        score_race(race_idx, race["entrants"], learning_rate, score_floor, negative_scaling, negative_scaling_cap, K_rate)
print(eval_pairs)

kt = sum(x[0] for x in eval_pairs) / (sum(x[1] for x in eval_pairs))
if args.trueskill:
    max_score = max(s.mu for s in user_scores.values())
else:
    max_score = max(user_scores.values())
print("recorded_only={}, learning rate={}, score_floor={}, negative_scaling={}, negative_scaling_cap={}, K_rate={}, kendall_tau={}, max_score={}".format(
    recorded_only, learning_rate, score_floor, negative_scaling, negative_scaling_cap, K_rate, kt, max_score))

histories = list(user_scores_history.items())
histories.sort(key=lambda x: len(x[1]), reverse=True)
# print(histories)
# print(sorted(user_scores.values()))

user_id_list = []
time_list = []
score_list = []
for user_id, h in histories[:15]:
    for race_idx, score in h:
        t = datetime.datetime.fromisoformat(race_list[race_idx]["ended_at"].replace('Z', '+00:00'))
        user_id_list.append(user_id)
        # time_list.append(t)
        time_list.append(race_idx)
        if args.trueskill:
            score_list.append(score.mu)
        else:
            score_list.append(score)
        
df = pd.DataFrame({"user_id": user_id_list, "time": time_list, "score": score_list})

sns.set_theme()
sns.relplot(df, kind="line", x="time", y="score", hue="user_id")
plt.show()
# 
# df2 = sns.load_dataset("dots")
# print(df2)
# print(df)
# print(sorted(user_scores.values()))