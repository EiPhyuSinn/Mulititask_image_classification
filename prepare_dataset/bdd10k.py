import os
import json

file_path = "/home/gwm-279/Downloads/10k_images_train/bdd100k/images/10k/train"
train_json = "/home/gwm-279/Downloads/bdd100k_labels_images_train.json"
val_json = "/home/gwm-279/Downloads/bdd100k_labels_images_val.json"
bdd_json = 'bdd10k.json'

# Load JSON files
with open(train_json, 'r') as f:
    train = json.load(f)
with open(val_json, 'r') as f:
    val = json.load(f)

# Convert image filenames to set for fast lookup
images_10k_set = set(os.listdir(file_path))

# Combine train and val for single loop
all_data = train + val

# Collect matching data
result = [
    {'name': data['name'], 'attributes': data['attributes']}
    for data in all_data
    if data['name'] in images_10k_set
]

# Save to JSON
with open(bdd_json, 'w') as f:
    json.dump(result, f, indent=4)

print('✅ All done')
