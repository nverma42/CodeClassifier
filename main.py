import ast
import networkx as nx
import matplotlib.pyplot as plt

def build_ast_graph(node, graph=None, parent=None):
    if graph is None:
        graph = nx.DiGraph()
    node_name = type(node).__name__
    node_id = f"{node_name}_{id(node)}"
    graph.add_node(node_id, label=node_name)
    if parent:
        graph.add_edge(parent, node_id)
    for child in ast.iter_child_nodes(node):
        build_ast_graph(child, graph, node_id)
    return graph

# This is the human or AI code that will be used to build the AST graph
# This graph will form the basis of the graph neural network.
code = """
x = a + b
"""
tree = ast.parse(code)
graph = build_ast_graph(tree)

# Get the labels from the node attributes
labels = nx.get_node_attributes(graph, 'label')

nx.draw(graph, with_labels=True, labels=labels, node_size=3000, node_color='lightblue', font_size=8, font_weight='bold')  # Visualizing AST
plt.show()
