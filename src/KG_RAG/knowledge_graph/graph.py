import networkx as nx
import matplotlib.pyplot as plt


def create_graph(triples, out):
    G = nx.DiGraph()

    for subj, rel, obj in triples:
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, label=rel)

    num_nodes = len(G.nodes)

    # Adjust k based on node count
    pos = nx.spring_layout(G, k=num_nodes/12, seed=1)

    edge_labels = nx.get_edge_attributes(G, 'label')

    # Dynamically adjust figure size
    # Keep it within reasonable bounds
    fig_size = min(max(10, num_nodes * 0.5), 25)
    plt.figure(figsize=(fig_size, fig_size))

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=2000, font_size=10, arrows=True)

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color='red',
        # Background to prevent overlap
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.title("Knowledge Graph for Edward Davis Wood Jr.")
    plt.savefig(out)


if __name__ == '__main__':
    # Sample triples
    triples = [
        ['Scott Derrickson', 'birthDate', 'July 16, 1966'], ['Scott Derrickson', 'nationality', 'America'], ['Scott Derrickson', 'residence', 'Los Angeles'], ['Scott Derrickson', 'residence', 'California'], ['Scott Derrickson', 'profession', 'Director'], ['Scott Derrickson', 'profession', 'Screenwriter'], ['Scott Derrickson', 'profession', 'Producer'], ['Sinister', 'director', 'Scott Derrickson'], ['The Exorcism of Emily Rose', 'director', 'Scott Derrickson'], ['Deliver Us From Evil', 'director', 'Scott Derrickson'], ['Doctor Strange', 'director',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            'Scott Derrickson'], ['Scott Derrickson', 'film', 'Sinister'], ['Scott Derrickson', 'film', 'The Exorcism of Emily Rose'], ['Scott Derrickson', 'film', 'Deliver Us From Evil'], ['Scott Derrickson', 'film', 'Doctor Strange'], ['Edward Davis Wood Jr.', 'nationality', 'American'], ['Edward Davis Wood Jr.', 'profession', 'Filmmaker'], ['Edward Davis Wood Jr.', 'profession', 'Actor'], ['Edward Davis Wood Jr.', 'profession', 'Writer'], ['Edward Davis Wood Jr.', 'profession', 'Producer'], ['Edward Davis Wood Jr.', 'profession', 'Director']
    ]

    triples_norm = [
        ['Scott Derrickson', 'birthDate', '1966-07-16'],
        ['Scott Derrickson', 'nationality', 'American'],
        ['Scott Derrickson', 'residence', 'Los Angeles, California, USA'],
        ['Scott Derrickson', 'profession', 'Film Director'],
        ['Scott Derrickson', 'profession', 'Screenwriter'],
        ['Scott Derrickson', 'profession', 'Film Producer'],
        ['Scott Derrickson', 'directorOf', 'Sinister'],
        ['Scott Derrickson', 'directorOf', 'The Exorcism of Emily Rose'],
        ['Scott Derrickson', 'directorOf', 'Deliver Us from Evil'],
        ['Scott Derrickson', 'directorOf', 'Doctor Strange'],
        ['Scott Derrickson', 'film', 'Sinister'],
        ['Scott Derrickson', 'film', 'The Exorcism of Emily Rose'],
        ['Scott Derrickson', 'film', 'Deliver Us from Evil'],
        ['Scott Derrickson', 'film', 'Doctor Strange'],
        ['Edward D. Wood Jr.', 'nationality', 'American'],
        ['Edward D. Wood Jr.', 'profession', 'Film Director'],
        ['Edward D. Wood Jr.', 'profession', 'Actor'],
        ['Edward D. Wood Jr.', 'profession', 'Screenwriter'],
        ['Edward D. Wood Jr.', 'profession', 'Film Producer']
    ]

    create_graph(triples, "graph.png")
    create_graph(triples_norm, "graph_norm.png")
