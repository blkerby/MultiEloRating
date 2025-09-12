import argparse
import logging
import os
import json
import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(
    prog='baseline_analysis',
    description='Baseline analysis using historical scores')
parser.add_argument("goal", help="Racetime goal (e.g., Map Rando S4)")
parser.add_argument("data_dir", help="Directory to load data from")
args = parser.parse_args()

data_path = os.path.join(args.data_dir, f"{args.goal}.json")
race_list = json.load(open(data_path, "r"))

MAX_PLACE = 10000

def eval_race(entrants):
    cnt_pair = 0
    cnt_incorrect = 0
    for i in range(len(entrants)):
        e1 = entrants[i]
        if e1["user_id"] is None:
            continue
        if e1["score"] is None:
            continue
        for j in range(i + 1, len(entrants)):
            e2 = entrants[j]
            if e2["user_id"] is None:
                continue
            if e2["score"] is None:
                continue
            if e1["place"] == e2["place"]:
                continue
            incorrect = (e1["place"] < e2["place"]) == (e1["score"] < e2["score"])
            if incorrect:
                cnt_incorrect += 1
            cnt_pair += 1
    return np.array(cnt_incorrect) / np.array(cnt_pair)
            
eval_fractions = []
for race in race_list:
    frac = eval_race(race["entrants"])
    if not np.isnan(frac):
        eval_fractions.append(float(frac))

print(np.mean(eval_fractions))
