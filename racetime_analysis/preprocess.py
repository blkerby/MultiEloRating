import argparse
import requests
import time
import logging
import os
import json

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(
    prog='baseline_analysis',
    description='Baseline analysis using recorded scores')
parser.add_argument("goal", help="Racetime goal (e.g., Map Rando S4)")
parser.add_argument("data_dir", help="Directory to load data from")
args = parser.parse_args()

data_dir = args.data_dir
goal = args.goal

def extract_data(race_data):
    entrants = []
    for e in race_data["entrants"]:
        if e["place"] is None:
            assert e["status"]["value"] in ["dnf", "dq"]
        entrants.append({
            "user_id": None if e["user"] is None else e["user"]["id"],
            "place": e["place"] if e["place"] is not None else 9999,
            "score": e["score"],
            "score_change": e["score_change"],
        })
    out = {
        "slug": race_data["slug"],
        "ended_at": race_data["ended_at"],
        "recorded": race_data["recorded"],
        "entrants": entrants,
    }
    return out

races_metadata_path = os.path.join(data_dir, "races.json")
race_metadata_list = json.load(open(races_metadata_path, "r"))
race_list = []
for race_metadata in race_metadata_list:
    if race_metadata["goal"]["name"] != goal:
        continue
    race_name = race_metadata["name"]
    race_slug = race_name.split("/")[1]
    race_path = f"{data_dir}/race/{race_slug}.json"
    if not os.path.exists(race_path):
        logging.warning(f"Skipping missing race: {race_name}")
        continue
    race_data = json.load(open(race_path, "r"))
    race_status = race_data["status"]["value"]
    open_time = race_data["opened_at"]
    if race_status == "finished":
        race_list.append(extract_data(race_data))
    else:
        logging.info(f"Skipping race: {race_name}, status {race_status}, opened {open_time}")

output_path = os.path.join(data_dir, f"{goal}.json")
race_list.sort(key=lambda r: r["ended_at"])
json.dump(race_list, open(output_path, "w"))
logging.info(f"Wrote {len(race_list)} races to {output_path}")