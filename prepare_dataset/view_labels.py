import json
from collections import Counter

# Path to your JSON file
json_file = "prepare_dataset/bdd10k.json"

with open(json_file, "r") as f:
    data = json.load(f)

# Counters for each attribute
weather_counter = Counter()
scene_counter = Counter()
time_counter = Counter()

for item in data:
    attrs = item["attributes"]
    weather_counter[attrs["weather"]] += 1
    scene_counter[attrs["scene"]] += 1
    time_counter[attrs["timeofday"]] += 1

# Sorted lists
weathers = sorted(weather_counter.keys())
scenes = sorted(scene_counter.keys())
times = sorted(time_counter.keys())

print("Weather categories:", weathers)
for w in weathers:
    print(f"  {w}: {weather_counter[w]} images")

print("\nScene categories:", scenes)
for s in scenes:
    print(f"  {s}: {scene_counter[s]} images")

print("\nTime of day categories:", times)
for t in times:
    print(f"  {t}: {time_counter[t]} images")
