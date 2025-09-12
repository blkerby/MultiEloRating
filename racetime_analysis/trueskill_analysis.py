import argparse
import logging
import os
import json
import numpy as np
import multi_elo_rating
import trueskill

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(
    prog='baseline_analysis',
    description='Baseline analysis using historical scores')
parser.add_argument("goal", help="Racetime goal (e.g., Map Rando S4)")
parser.add_argument("data_dir", help="Directory to load data from")
args = parser.parse_args()

data_path = os.path.join(args.data_dir, f"{args.goal}.json")
race_list = json.load(open(data_path, "r"))

trueskill_env = trueskill.TrueSkill(backend='mpmath')
default_rating = trueskill_env.create_rating()
    
user_scores = {}

# Evaluate using kendall-tau correlation:
def eval_race(entrants):
    cnt_pair = 0
    cnt_incorrect = 0
    for i in range(len(entrants)):
        e1 = entrants[i]
        user1 = e1["user_id"]
        if user1 is None:
            continue
        score1 = user_scores.get(user1, default_rating).mu
        for j in range(i + 1, len(entrants)):
            e2 = entrants[j]
            user2 = e2["user_id"]
            if user2 is None:
                continue
            score2 = user_scores.get(user2, default_rating).mu
            if e1["place"] == e2["place"]:
                continue
            if score1 == score2:
                cnt_incorrect += 0.5
            else:
                incorrect = (e1["place"] < e2["place"]) == (score1 < score2)
                if incorrect:
                    cnt_incorrect += 1
            cnt_pair += 1
    return np.array(cnt_incorrect) / np.array(cnt_pair)

def score_race(entrants):
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
    
    # rating_grad = multi_elo_rating.get_rating_gradient(rating_list, rank_list)
    rating_list = [(x,) for x in rating_list]
    new_ratings = trueskill_env.rate(rating_list, rank_list)
    new_ratings = [x[0] for x in new_ratings]

    for i in range(len(user_id_list)):
        user_id = user_id_list[i]
        new_rating = new_ratings[i]
        user_scores[user_id] = new_rating

recorded_only = True

eval_fractions = []
for race in race_list:
    if recorded_only and not race["recorded"]:
        continue
    frac = eval_race(race["entrants"])
    if not np.isnan(frac):
        eval_fractions.append(float(frac))
        score_race(race["entrants"])

print("recorded_only={}, kendall_tau={}".format(recorded_only, np.mean(eval_fractions)))
