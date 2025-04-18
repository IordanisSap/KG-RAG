import json
import string
import re
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

with open("gold.jsonl", "r") as gold_file, open("predictions.jsonl", "r") as pred_file:
    gold_data = {json.loads(line)["id"]: json.loads(line) for line in gold_file}
    pred_data = {json.loads(line)["id"]: json.loads(line) for line in pred_file}

total = len(gold_data)
f1_total = 0
em_total = 0

for id_, gold_item in gold_data.items():
    gold_answer = gold_item["answer"]
    pred_answer = pred_data[id_]["predicted_answer"]
    f1_total += f1_score(pred_answer, gold_answer)
    em_total += exact_match_score(pred_answer, gold_answer)

f1 = f1_total / total * 100
em = em_total / total * 100

print(f"Exact Match (EM): {em:.2f}%")
print(f"F1 Score: {f1:.2f}%")