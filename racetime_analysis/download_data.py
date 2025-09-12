import argparse
import requests
import time
import logging
import os
import json

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(
    prog='download',
    description='Download race data from racetime.gg')
parser.add_argument("category", help="Racetime category (e.g., smr)")
parser.add_argument("goal", help="Racetime goal (e.g., Map Rando S4)")
parser.add_argument("data_dir", help="Directory to load/store data in")
parser.add_argument("--request_delay", type=int, default=5, help="Delay (in seconds) between requests to Racetime")
args = parser.parse_args()

data_dir = args.data_dir
os.makedirs(os.path.join(data_dir, "race"), exist_ok=True)
races_metadata_path = os.path.join(data_dir, "races.json")
race_list = json.load(open(races_metadata_path, "r"))

base_url = "https://racetime.gg"
category = args.category

filtered_race_list = [race for race in race_list if race["goal"]["name"] == args.goal]
num_races = len(filtered_race_list)
for i, race in enumerate(filtered_race_list):
    if race["goal"]["name"] != args.goal:
        continue
    
    race_name = race["name"]
    race_slug = race_name.split("/")[1]
    race_path = f"{data_dir}/race/{race_slug}.json"
    race_data = None
    if os.path.exists(race_path):
        race_data = json.load(open(race_path, "r"))
        if race_data["status"]["value"] in ["finished", "cancelled"]:
            logging.info(f"Skipping existing race: {race_name} ({i + 1}/{num_races})")
            continue

    url = f"{base_url}/{category}/{race_slug}/data"
    logging.info(f"Downloading {url}")
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Request failed with status {response.status_code}: {response.text()}")
    race_data = response.json()
    race_status = race_data["status"]["value"]
    open_time = race_data["opened_at"]
    logging.info(f"Success: race {race_name}, status {race_status}, opened {open_time} ({i + 1}/{num_races})")
    json.dump(race_data, open(race_path, "w"))
    logging.info(f"Waiting {args.request_delay} seconds")
    time.sleep(args.request_delay)
