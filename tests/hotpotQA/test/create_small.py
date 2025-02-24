import json

file_path = "../hotpot_dev_distractor_v1.json"  # Change this to your JSON file's path
out_path = "../hotpot_dev_distractor_v1_small.json"  # Change this to your JSON file's path


with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)  # Load JSON data
    print("JSON data loaded successfully!")
    print(len(data))
    out_data = [data[i] for i in range(0,100)]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out_data, indent=4))
    

