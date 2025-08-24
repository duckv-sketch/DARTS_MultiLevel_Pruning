
import networkx as nx
import matplotlib.pyplot as plt

def plot_cell(genotype, cell_type='normal'):
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout

    G = nx.DiGraph()
    edges = getattr(genotype, cell_type)

    steps = len(edges) // 2
    for i in range(steps):
        for j in range(2):
            op, idx = edges[2 * i + j]
            src = f"input_{idx}" if idx < 2 else f"{idx - 2}"
            dst = f"{i}"
            G.add_edge(src, dst, label=op)

    # Use neato layout for better spacing
    pos = graphviz_layout(G, prog='neato')

    # Manually scale node positions to spread out
    for k in pos:
        x, y = pos[k]
        pos[k] = (x * 2.5, y * 2.5)

    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(24, 16))  # Large canvas
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='skyblue',
        node_size=3000,
        font_size=14,
        arrows=True,
        arrowsize=25,
        width=2.5
    )
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='red',
        font_size=13,
        label_pos=0.5,
        rotate=False
    )

    plt.title(f"{cell_type.capitalize()} Cell Architecture", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{cell_type}_cell_clear.png")
    plt.show()
