import copy
import json

categories = [
              {"name": "aeroplane", "id": 1, "trainId": 0},
              {"name": "bicycle", "id": 2, "trainId": 1},
              {"name": "bird", "id": 3, "trainId": 2},
              {"name": "boat", "id": 4, "trainId": 3},
              {"name": "bottle", "id": 5, "trainId": 4},
              {"name": "bus", "id": 6, "trainId": 5},
              {"name": "car", "id": 7, "trainId": 6},
              {"name": "cat", "id": 8, "trainId": 7},
              {"name": "chair", "id": 9, "trainId": 8},
              {"name": "cow", "id": 10, "trainId": 9},
              {"name": "diningtable", "id": 11, "trainId": 10},
              {"name": "dog", "id": 12, "trainId": 11},
              {"name": "horse", "id": 13, "trainId": 12},
              {"name": "motorbike", "id": 14, "trainId": 13},
              {"name": "person", "id": 15, "trainId": 14},
              {"name": "potted plant", "id": 16, "trainId": 15},
              {"name": "sheep", "id": 17, "trainId": 16},
              {"name": "sofa", "id": 18, "trainId": 17},
              {"name": "train", "id": 19, "trainId": 18},
              {"name": "tvmonitor", "id": 20, "trainId": 19}]

categories_seen = copy.deepcopy(categories[:15])

categories_unseen = copy.deepcopy(categories[15:])
for index, item in enumerate(categories_unseen):
    item["trainId"] = index

with open(r'datasets/VOCZERO/all_classnames.json', 'w') as f_out:
    all_categories_json = []
    for cat in categories:
        all_categories_json.append(cat["name"])
    json.dump(all_categories_json, f_out)

with open(r'datasets/VOCZERO/seen_classnames.json', 'w') as f_out:
    seen_categories_json = []
    for cat in categories_seen:
        seen_categories_json.append(cat["name"])
    json.dump(seen_categories_json, f_out)

with open(r'datasets/VOCZERO/unseen_classnames.json', 'w') as f_out:
    unseen_categories_json = []
    for cat in categories_unseen:
        unseen_categories_json.append(cat["name"])
    json.dump(unseen_categories_json, f_out)