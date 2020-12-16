import pickle
import networkx as nx

from collections import defaultdict

class Loud:

    def __init__(self,
        graph_path='causality_graph.pickle',
        page_rank_alpha=0.85, 
        page_rank_max_iter=10000,
        kpi_limit=20
    ):
        with open(graph_path, 'rb') as f:
            self.causality_graph = pickle.load(f)
        
        self.pagerank_alpha = page_rank_alpha
        self.max_pagerank_iter = page_rank_max_iter
        self.kpi_limit = kpi_limit
            

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

    def root_cause_detection(self, kpis):
        """
            Main function for calculation of root cause
            kpis : collection of pairs (host, kpi)
            returns : most likely root cause
        """
        subgraph = self.get_subgraph(kpis)
        
        scoring = sorted(tuple(self.apply_pagerank(subgraph).items()), key=lambda x: -x[1])

        scoring = scoring[:self.kpi_limit] # limits to a number of ocurring KPIs
        
        host_kpis = defaultdict(list)
        
        scoring = tuple(map(lambda x: x[0], scoring))
        for val in scoring:
            host,kpi = val.split(':')
            host_kpis[host].append(kpi)
        
        #FIXME paper returns only the most likely host ?
        return [(host, kpi) for host in host_kpis for kpi in host_kpis[host]] 


if __name__ == '__main__':
    model = Loud()

    pairs = (('docker_003', 'container_cpu_used'), ('docker_004','container_cpu_used'),
                        ('db_009', 'On_Off_State'), ('redis_012','expired_keys'))
    subgraph = model.get_subgraph(pairs)

    vector = model.apply_pagerank(subgraph)

    print(model.root_cause_detection(pairs))