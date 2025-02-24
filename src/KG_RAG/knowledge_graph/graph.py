import networkx as nx
import matplotlib.pyplot as plt

# Sample triples
triples = [
  ["Scott Derrickson", "born on", "July 16, 1966"],
  ["Scott Derrickson", "is from", "America"],
  ["Scott Derrickson", "lives in", "Los Angeles"],
  ["Scott Derrickson", "lives in", "California"],
  ["Scott Derrickson", "is", "American director"],
  ["Scott Derrickson", "is", "screenwriter"],
  ["Scott Derrickson", "is", "producer"],
  ["Sinister", "directed by", "Scott Derrickson"],
  ["The Exorcism of Emily Rose", "directed by", "Scott Derrickson"],
  ["Deliver Us From Evil", "directed by", "Scott Derrickson"],
  ["Doctor Strange", "directed by", "Scott Derrickson"],
  ["Scott Derrickson", "best known for", "horror films"],
  ["Scott Derrickson", "worked on", "Marvel Cinematic Universe"],
  ["Edward Davis Wood Jr.", "was", "American"],
  ["Edward Davis Wood Jr.", "is", "filmmaker"],
  ["Edward Davis Wood Jr.", "is", "actor"],
  ["Edward Davis Wood Jr.", "is", "writer"],
  ["Edward Davis Wood Jr.", "is", "producer"],
  ["Edward Davis Wood Jr.", "is", "director"],
  ["Edward Davis Wood Jr.", "born on", "October 10, 1924"],
  ["Edward Davis Wood Jr.", "died on", "December 10, 1978"]
]

# Create a directed graph
G = nx.DiGraph()

# Add triples to the graph
for subj, rel, obj in triples:
    G.add_node(subj)
    G.add_node(obj)
    G.add_edge(subj, obj, label=rel)

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # For consistent layout
edge_labels = nx.get_edge_attributes(G, 'label')

plt.figure(figsize=(20, 20))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title("Knowledge Graph for Edward Davis Wood Jr.")
plt.savefig("dummy_name.png")