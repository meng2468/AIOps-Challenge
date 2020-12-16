import pickle
import networkx as nx

class Loud:

    def __init__(self, graph_path='causality_graph.pickle'):
        with open(graph_path, 'rb') as f:
            self.causality_graph = pickle.load(f)

    def get_subgraph(self, pairs):
        keys = list(map(lambda x: ':'.join(x), pairs))
        subgraph = nx.DiGraph()
        subgraph.add_nodes_from(keys)
        for key in keys:
            edges = [(key, dst) for dst in filter(lambda x: x in keys, self.causality_graph.successors(key))]
            for edge in edges:
                subgraph.add_edge(edge[0], edge[1], **self.causality_graph.get_edge_data(edge[0], edge[1]))
        
        return subgraph
                
        


if __name__ == '__main__':
    model = Loud()

    pairs = (('docker_003', 'container_cpu_used'), ('docker_004', 'container_cpu_used'), ('db_009', 'On_Off_State'))
    subgraph = model.get_subgraph(pairs)
