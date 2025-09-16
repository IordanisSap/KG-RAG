import os

questions_path = os.path.join(os.path.dirname(__file__), "WC-P1.txt")
pred_path = os.path.join(os.path.dirname(__file__), "WC-P1-pred-raw.txt")
correct = 0
wrong= 0

with open("wrong2.txt", 'w') as outFile:
    with open(pred_path, 'r') as predFile:
        with open(questions_path, 'r') as qFile:
            for line_q, line_pred in zip(qFile, predFile):
                question, ansData = line_q.split("?")
                answer = ansData.strip().split("\t")[0]
                valid_answers = ansData.strip().split("\t")[2][:-1].split("/")
                valid_answers = [ans.lower() for ans in valid_answers]
                if (line_pred.rstrip().replace(" ", "_").lower().split(",")[0] in valid_answers):
                    correct+=1
                else:
                    print(line_pred.rstrip().lower().split(",")[0], valid_answers[0])
                    outFile.write(line_q)
                    wrong+=1

print(correct)
print(wrong)

