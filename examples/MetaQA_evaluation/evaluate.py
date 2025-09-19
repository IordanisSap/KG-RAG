import os
import re
import string

def normalize(s: str) -> str:
    s = s.strip().lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s)
    return s

def to_set(items):
    return {normalize(x) for x in items if x.strip()}

def prf1(gold_set, pred_set):
    tp = len(gold_set & pred_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp

gold_path = os.path.join(os.path.dirname(__file__), '1-hop', 'qa_test.txt')
pred_path = os.path.join(os.path.dirname(__file__), '1-hop_pred', 'pred-non-verbalized-gemma.txt')

total_tp = total_pred = total_gold = 0
macro_f1s = []
macro_precisions = []
macro_recalls = []
correct_count = 0
exact_match_count = 0

with open(gold_path, 'r', encoding='utf-8') as benchFile, open(pred_path, 'r', encoding='utf-8') as pred_file:
    for i, (gold_line, pred_line) in enumerate(zip(benchFile, pred_file), 1):
        question, answers_str = gold_line.rstrip("\n").split("\t")
        answers = answers_str.split("|")
        preds = pred_line.rstrip("\n").split("|")

        gset = to_set(answers)
        pset = to_set(preds)

        # Top-1 prediction
        top_pred = normalize(preds[0]) if preds else None
        if top_pred and top_pred in gset:
            correct_count += 1

        # Exact Match
        if gset == pset:
            exact_match_count += 1

        # Precision, Recall, F1
        p, r, f1, tp = prf1(gset, pset)
        macro_f1s.append(f1)
        macro_precisions.append(p)
        macro_recalls.append(r)

        total_tp += tp
        total_pred += len(pset)
        total_gold += len(gset)

        print(f"[{i}] F1={f1:.3f}  P={p:.3f}  R={r:.3f}  EM={'✓' if gset == pset else '✗'}  gold={sorted(gset)}  pred={sorted(pset)}")

# Micro scores
micro_p = total_tp / total_pred if total_pred else 0.0
micro_r = total_tp / total_gold if total_gold else 0.0
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

# Macro scores
macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
macro_p = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
macro_r = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0

# Accuracy and Exact Match
accuracy = correct_count / i if i else 0.0
exact_match = exact_match_count / i if i else 0.0

# Final report
print("\nFinal Scores:")
print(f"Accuracy (top-1 match):                   {accuracy:.4f}")
print(f"Exact Match (unordered full match):       {exact_match:.4f}")
print(f"Macro Precision:                          {macro_p:.4f}")
print(f"Macro Recall:                             {macro_r:.4f}")
print(f"Macro F1:                                 {macro_f1:.4f}")
print(f"Micro Precision:                          {micro_p:.4f}")
print(f"Micro Recall:                             {micro_r:.4f}")
print(f"Micro F1:                                 {micro_f1:.4f}")
