from KG_RAG.pipeline import RAGAgent
# from KG_RAG.knowledge_graph.nlp import lemmatize_word
import json

        
if __name__ == "__main__":
    test_case = {
        "_id": "5a8b57f25542995d1e6f1371",
        "answer": "yes",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "supporting_facts": [
            [
                "Scott Derrickson",
                0
            ],
            [
                "Ed Wood",
                0
            ]
        ],
        "context": [
            [
                "Scott Derrickson",
                [
                    "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
                    " He lives in Los Angeles, California.",
                    " He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\""
                ]
            ],
            [
                "Ed Wood",
                [
                    "Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director."
                ]
            ],
        ],
        "type": "comparison",
        "level": "hard"
    }
    
    dict = {}    

    agent = RAGAgent()
    for context in test_case["context"]:
        dict[context[0]] = {}
        for index,sentence in enumerate(context[1]):
            dict[context[0]][index] = []
        sentences = "".join(context[1])
        triples = agent.generate_triples(sentences)
        print(triples)
        parsed_triples = json.loads(triples)["triples"]
        for triple in parsed_triples:
            for index, sentence in enumerate(context[1]):
                if (sentence.rfind(triple[2]) > -1): 
                    dict[context[0]][index].append(triple)
    # print(dict)
    for key,paragraph in dict.items():
        for index, sentence in dict[key].items():
            print("{0}[{1}]: {2}".format(key,index,sentence))
    print(print("Question: {0}".format(agent.generate_triples(test_case["question"]))))
    print(agent.get_similarity('filmDirector', 'Film_director'))
