from pyvis.network import Network
import networkx as nx

def visualize_graph(G, traversal_nodes):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")

    # Convert traversal_nodes to a set of integers for efficient lookup
    traversal_nodes_set = set(int(node) for node in traversal_nodes)

    # Add nodes to the network
    for node in G.nodes():
        node_id = int(node)
        color = 'red' if node_id in traversal_nodes_set else 'blue'
        node_label = str(node_id)
        node_title = G.nodes[node]['text']

        net.add_node(node_id, label=node_label, title=node_title, color=color)

    # Add edges to the network
    for edge in G.edges():
        source = int(edge[0])
        target = int(edge[1])
        net.add_edge(source, target)

    net.show_buttons(filter_=['physics'])

    return net
