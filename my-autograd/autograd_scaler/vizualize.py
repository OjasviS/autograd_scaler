from graphviz import Digraph
from value import Value

def _trace(root):
    edges = set()
    nodes = set()
    def dfs(node):
        
        if node not in nodes:
            nodes.add(node)
            for parent in node._parent_operands:
                edge = (parent, node)
                edges.add(edge)
                dfs(parent)
    dfs(root)
    return nodes, edges
    
def draw(root):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})
    nodes, edges = _trace(root)

    for n in nodes:
        # Node label: value data
        dot.node(name=str(id(n)), label=f" data = {n.data} | label = {n.label} | grad = {n.grad}", shape = 'record')

        # If this Value was created by an operation, draw an op node
        if n._creator_op:
            op_node = str(id(n)) + n._creator_op
            dot.node(op_node, label=n._creator_op, shape='circle')
            dot.edge(op_node, str(id(n)))

            for parent in n._parent_operands:
                dot.edge(str(id(parent)), op_node)

    return dot    