from fastcoref import FCoref

model = FCoref(device='cuda:0')

text = 'We are so happy to see you, George, using our coref package. This package is very fast and you and your dog are the best!'


preds = model.predict(
   texts=[text]
)

print(preds[0].get_clusters(as_strings=False))


clusters = preds[0].get_clusters(as_strings=False)

def replaceRefs(text, clusters):
    for cluster in clusters:
        print("Pairs: {0}".format(",".join([text[pair[0]:pair[1]] for pair in cluster])))
        # print("'{1}' will be replaced by '{0}'".format(text[cluster[0][0]:cluster[0][1]],text[cluster[1][0]:cluster[1][1]]))
    
    
replaceRefs(text,clusters)

# import spacy

# nlp = spacy.load("en_core_web_sm")

# sentence = "We are so happy to see you, George, using our coref package. This package is very fast and you are the best!"

# doc = nlp(sentence)


# resolved_text = ""
# for token in doc:
  
#     repres = doc._.coref_chains.resolve(token)
#     print(repres)
#     if repres:
#         resolved_text += " " + " and ".join([t.text for t in repres])
#     else:
#         resolved_text += " " + token.text
    
# print(resolved_text)
