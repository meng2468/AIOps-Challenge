import pickle
import networkx as nx


class Loud:

    def __init__(self,
        graph_path='causality_graph.pickle',
        page_rank_alpha=0.85, 
        page_rank_max_iter=10000,
    ):
        with open(graph_path, 'rb') as f:
            self.causality_graph = pickle.load(f)
        
        self.pagerank_alpha = page_rank_alpha
        self.max_pagerank_iter = page_rank_max_iter

            

    def get_subgraph(self, pairs):
        keys = list(map(lambda x: ':'.join(x), pairs))
        subgraph = nx.DiGraph()
        subgraph.add_nodes_from(keys)
        for key in keys:
            edges = [(key, dst) for dst in filter(
                lambda x: x in keys, self.causality_graph.successors(key))]
            for edge in edges:
                subgraph.add_edge(
                    edge[0], edge[1], **self.causality_graph.get_edge_data(edge[0], edge[1]))

        return subgraph

    def apply_pagerank(self, graph):
        # personalization value is the average weight of the outgoing edges
        # FIXME does this make sense? maybe we want uniformly and the edges handle everything
        # personalization = { node:
        #         sum(map(lambda x: x[2]['weight'], graph.out_edges(node, data=True))) / len(graph.out_edges(node)) 
        #             if len(graph.out_edges(node)) != 0 
        #             else 0
        #         for node in graph.nodes
        # }
        
        return nx.pagerank(
            graph,
            personalization=None, # FIXME above
            alpha = self.pagerank_alpha,
            max_iter = self.max_pagerank_iter
        )


if __name__ == '__main__':
    model = Loud()

    pairs = (('docker_003', 'container_cpu_used'), ('docker_004','container_cpu_used'),
                        ('db_009', 'On_Off_State'), ('redis_012','expired_keys'))
    subgraph = model.get_subgraph(pairs)

    vector = model.apply_pagerank(subgraph)
