import argparse
import requests
import time
import logging
import os
import json

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(
    prog='download',
    description='Download race metadata from racetime.gg')
parser.add_argument("category", help="Racetime category (e.g., smr)")
parser.add_argument("data_dir", help="Directory to store data in")
parser.add_argument("--request_delay", type=int, default=5, help="Delay (in seconds) between requests to Racetime")
parser.add_argument("--page_limit", type=int, default=None, help="Maximum number of pages to download")
args = parser.parse_args()

base_url = "https://racetime.gg"
category = args.category

# Keep races in a dict, keyed by race name, in order to deduplicate them:
# duplicates could arise if new races are opened in the middle of downloading,
# since this would shift how the races align to pages.
race_dict = {}
next_page = 1
per_page = 100
while True:
    url = f"{base_url}/{category}/races/data?page={next_page}&per_page={per_page}"
    logging.info(f"Downloading {url}")
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Request failed with status {response.status_code}: {response.text()}")
    response_json = response.json()
    num_pages = response_json["num_pages"]
    for race in response_json["races"]:
        race_dict[race["name"]] = race
    race_count = response_json["count"]
    logging.info(f"Success: page {next_page}/{num_pages}, race count {race_count}")

    next_page += 1
    if next_page > num_pages:
        break
    if args.page_limit is not None and next_page > args.page_limit:
        break
    logging.info(f"Waiting {args.request_delay} seconds")
    time.sleep(args.request_delay)

race_list = list(race_dict.values())
output_path = os.path.join(args.output_dir, "races.json")
logging.info(f"Writing race metadata to {output_path}")
os.makedirs(args.output_dir, exist_ok=True)
json.dump(race_list, open(output_path, "w"))
