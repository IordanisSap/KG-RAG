import os

questions_path = os.path.join(os.path.dirname(__file__), "WC-P1.txt")
pred_path = os.path.join(os.path.dirname(__file__), "WC-P1-pred-verbalized-normal.txt")

correct = 0
wrong = 0
total = 0

with open(pred_path, 'r', encoding='utf-8') as predFile, open(questions_path, 'r', encoding='utf-8') as qFile:
    for line_q, line_pred in zip(qFile, predFile):
        # Keep your original parsing EXACTLY
        question, ansData = line_q.split("?")
        answer = ansData.strip().split("\t")[0]
        valid_answers = ansData.strip().split("\t")[2][:-1].split("/")  # keep [:-1]
        valid_answers = [ans.lower() for ans in valid_answers]

        top1 = line_pred.rstrip().replace(" ", "_").lower().split(",")[0]

        total += 1
        if top1 in valid_answers:
            correct += 1
        else:
            wrong += 1
            # print(line_pred.rstrip().lower().split(",")[0], valid_answers[0])

hits_at_1 = correct / total if total else 0.0
print(f"hits@1: {hits_at_1:.4f}  ({correct}/{total})")
