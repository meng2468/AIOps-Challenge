import pandas as pd
from sklearn.cluster import Birch
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from collections import defaultdict

import os
import re

import math

from pprint import pprint

from scipy.stats import pearsonr

class MicroRCA:

    def __init__(self,  
        smoothing_window=12, 
        min_periods=1, 
        branching_factor=50, 
        ad_threshold=0.045, 
        weights_alpha=0.55, 
        page_rank_alpha=0.85, 
        page_rank_max_iter=10000,
        model_checkoint='',
        debug=False,
        pickle_folder=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/'),
    ):   
        # mean and std deviation of training duration
        self.smoothing_window = smoothing_window
        self.min_periods = min_periods

        self.branching_factor = branching_factor 
        self.ad_threshold = ad_threshold
        
        self.weights_alpha = weights_alpha

        self.page_rank_alpha = page_rank_alpha
        self.page_rank_max_iter = page_rank_max_iter

        self.debug = debug

        
        files = os.listdir(pickle_folder)
        self.detectors = {}
        
        for file in files:
            m = re.search(r'(.+)_(.+)_cluster_model.pickle', file)        
            if m:
                groups = m.groups()
                key = groups[0] + groups[1]
                with open(pickle_folder + file, 'rb') as f:
                    self.detectors[key] = pickle.load(f)


    def detect(self, traces_df, kpis, visualize=False):
        # Parse the traces and kpis
        parsed_traces = self.parse_traces(traces_df)
        
        # FIXME possible case where system doesn't answer for a long time and wasn't called

        #check for anomaly
        # 1 - find outlier in elapsed
        #   1.1 microRCA

        traces = self.get_anomalous_traces(traces_df)
        # print(traces, len(traces))
        # print(self.detectors['osb_001OSB'].anomaly_index)
        # print(parsed_traces[traces[0]].head())
        # print('Unique services in trace:', len(parsed_traces[traces[0]]['serviceName'].unique()))

        # Should we iterate here over each trace in traces or consider them all together?
        # for trace_id in traces:
        #   trace = parsed_traces[trace_id]
        #   DG = self.trace_graph(trace)
        #   ... [Perform the rest of the steps]

        # TODO Build attribute graph 
        # Hosts + Service
        # Each service connects to all the services it communicates with and all hosts it connects to (no need to differentiate!)
        DG = nx.DiGraph()
        for trace in traces:
            DG = self.trace_graph(parsed_traces[trace], DG)
        
        if visualize:
            print(DG.nodes(data=True), len(DG.nodes()))

            plt.figure(figsize=(9,9))
            pos = nx.spring_layout(DG)
            nx.draw(DG, pos, with_labels=True, cmap=plt.get_cmap('jet'), node_size=0, arrows=True)

            # nx.draw_networkx_nodes(DG, pos, nodelist=hosts, node_color="r", node_size=1500)
            # nx.draw_networkx_nodes(DG, pos, nodelist=services, node_color="b", node_size=500)
            nx.draw_networkx_edges(DG, pos, width=1.0, alpha=0.5)

            labels = nx.get_edge_attributes(DG, 'weight')
            nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels)
            plt.show()

        # Extract anomalous subgraph
        anomaly_DG, anomalous_edges = self.get_anomalous_graph(DG, traces, parsed_traces, traces_df)

        # TODO Faulty service localization
        # Update weights of anomalous graph
        #           Use cases from the paper
        # Get personalization vector (Transition Probability Matrix)
        # Reverse the service-service edges
        # Apply pagerank
        parsed_kpis = self.parse_kpis(kpis)
        
        result = self.get_fault_service(anomaly_DG, anomalous_edges, traces_df, parsed_kpis)
        # result = list(map(lambda x: x[0], filter(lambda x: x[1] > 0, result)))

        # hosts = []
        # for v in result:
        #     hosts.append(set(map(lambda x: x[1], filter(lambda x: anomaly_DG.nodes[x[1]]['type']=='host', anomaly_DG.out_edges(v)))))

        return result

    def get_anomalous_traces(self, tracelist):
        """
        tracelist - pd dataframe with traces

        returns: iterable of traceids that are anomalous
        """
        # get roots (always OSB)
        traces = tracelist[tracelist['callType'] == 'OSB']
        model = self.detectors['osb_001OSB']

        predictions = model.predict(traces['elapsedTime'].values.reshape(-1,1))
        indexes = np.where(predictions != model.anomaly_index)
        
        return traces.iloc[indexes]['traceId'].unique()

    def parse_traces(self, traces):
        traces = dict(tuple(traces.sort_values('traceId').groupby('traceId')))
        # for key in traces:
        #     df = traces[key].sort_values('startTime')
        #     print(df['elapsedTime'].iloc[0], df['traceId'].iloc[0], sep='\t')
        # i = 1
        return traces 

    def parse_kpis(self, kpis):
        return dict(tuple(kpis.groupby('cmdb_id')))


    def anomalus_subgraph(self, DirectedGraph, anomalies, ):
        pass

    def trace_graph(self, trace, prev_graph, visualize=False):
        DG = nx.DiGraph(prev_graph)
        
        hosts = trace['cmdb_id'].unique()
        services = trace['serviceName'].unique()

        # print(30*'-')
        # print(hosts, len(hosts))
        # print(services, len(services))
        # print(30*'-')

        # Add nodes to the graph
        for node in hosts:
            DG.add_node(node, type='host')
        
        for node in services:
            DG.add_node(node, type='service')
  
        # Add edges to the graph
        for _, row in trace.iterrows():
            parent = trace[trace['id'] == row['pid']]['serviceName']
            service = row['serviceName']
            host = row['cmdb_id']
        
            # Parent service to current service
            if(len(parent)): # Parent may be empty
                DG.add_edge(parent.values[0], service)
         
            # Current service to its host
            DG.add_edge(service, host)

        return DG

    def get_anomalous_graph(self, graph, anomalousids, traces, trace_df):
        anomalous_nodes = set()
        
        def detect_nodes(row):
            # {k -> v} k = serviceName + callType 
            model = self.detectors[row.loc['serviceName'] + row.loc['callType']]
            prediction = model.predict(np.array(row['elapsedTime']).reshape(-1,1))
            if prediction != model.anomaly_index: # anomaly_index is inverted. It signals the normal index!!!
                anomalous_nodes.add(row.loc['serviceName'])

        # find anomalous nodes
        for id in anomalousids:
            trace = traces[id]
            trace.apply(detect_nodes, axis=1)

        anomalous_graph = nx.DiGraph()
        
        anomalous_edges = []
        for node in anomalous_nodes:
            changed = False
            #if any(map(lambda n: n in anomalous_nodes, graph.predecessors(node))):
            for parent in graph.predecessors(node):
                # Avoid self loops (in CSF inside and outside span) 
                if parent == node:
                    continue

                if parent in anomalous_nodes:
                    changed = True
                    anomalous_edges.append((parent, node))
            
            if changed:
                anomalous_graph.add_node(node, status='anomaly', type=graph.nodes[node]['type'])

        anomalous_nodes = list(anomalous_graph.nodes)
        for node in anomalous_nodes:
            for n in graph.predecessors(node):
                if n not in anomalous_graph.nodes:
                    anomalous_graph.add_node(n, status='normal', **graph.nodes[n])
            
            for n in graph.successors(node):
                if n not in anomalous_graph.nodes:
                    anomalous_graph.add_node(n, status='normal', **graph.nodes[n])
            
            anomalous_graph.add_edges_from(list(map(lambda x: x[::-1], graph.in_edges(node))))
            anomalous_graph.add_edges_from(graph.out_edges(node))

        
        for node in anomalous_graph.nodes:
            if anomalous_graph.nodes[node]['type'] == 'service':
                avg = trace_df[trace_df['serviceName'] == node]['elapsedTime'].mean()
                anomalous_graph.nodes[node]['rt'] = avg

        return anomalous_graph, anomalous_edges

    def get_max_correlation(self, times, host_kpi_df):
        """ Returns absolute value of correlation """
        max_corr = 0
        for k in host_kpi_df['name'].unique():
            serie = host_kpi_df[host_kpi_df['name'] == k]['value']
            if len(serie.index) == 1 or serie.var() == 0:
                continue
            
            corr = serie.corr(times) # FIXME divide by zero sometimes?

            if math.isnan(corr): # in case kpis don't vary
                corr = 0
            
            #l.append(abs(corr))
            corr = abs(corr)
            max_corr = max(corr, max_corr)

        return max_corr

    def get_fault_service(self, graph, anomalous_edges, traces, kpis):
        for v in graph.nodes:
            if graph.nodes[v]['status'] != 'anomaly':
                continue
            in_val = 0
            for edge in graph.in_edges(v):
                src, _ = edge
                weight = self.weights_alpha if edge in anomalous_edges else traces[traces['serviceName'] == v]['elapsedTime'].corr(traces[traces['serviceName'] == src]['elapsedTime'])
                if math.isnan(weight): # in case kpis don't vary
                    weight = 0

                in_val += weight
                new_edge = (src, v, {'weight': weight})
            
                nx.set_edge_attributes(graph, {(src, v) : {'weight' : weight}})
            
            in_val /= graph.in_degree(v)

            for edge in graph.out_edges(v):
                _, dst = edge
                if (edge in anomalous_edges):
                    continue
                if graph.nodes[dst]['type'] == 'service':
                    val = traces[traces['serviceName'] == v]['elapsedTime'].corr(traces[traces['serviceName'] == dst]['elapsedTime'])
                    if math.isnan(val): # in case kpis don't vary
                        val = 0
                else: # host
                    kpi = kpis[dst]

                    times = traces[traces['serviceName'] == v]['elapsedTime']
                    max_corr = self.get_max_correlation(times, kpi)
                    val = in_val * max_corr
                nx.set_edge_attributes(graph, {(v, dst) : {'weight' : val}})
        
        personalization = {}
        for v in graph.nodes:
            if graph.nodes[v]['status'] != 'anomaly':
                continue    
            
            #get avg weight
            vals = [graph.get_edge_data(*edge)['weight'] for edge in graph.out_edges(v)] #all 
            if 'csf' in v:
                print(v, vals)
            avg = sum(vals) / len(vals)


            # get max correlation value
            max_corr = 0

            times = traces[traces['serviceName'] == v]['elapsedTime']

            keys = filter(lambda x: graph.nodes[x[1]]['type']=='host', graph.out_edges(v))
            for key in keys:
                kpi_df = kpis[key[1]]
                val = self.get_max_correlation(times, kpi_df)
                max_corr = max(max_corr, val)
            
            val = avg * max_corr
            personalization[v] = val / graph.degree(v) # why do they do this in the original code?
                
        reversed_graph = graph.reverse(copy=True)

        scores = nx.pagerank(
            reversed_graph, 
            alpha=self.page_rank_alpha,
            personalization=personalization,
            max_iter=self.page_rank_max_iter
        )
        scores = sorted(filter(lambda x: x[1] > 0, scores.items()), key=lambda x: x[1], reverse=True)
        print(scores)
        hosts = list(filter(lambda x: x[1]['type'] == 'host', graph.nodes(data=True)))
        
        max_score = max(map(lambda x: x[1], scores))
        result = set(map(lambda x: x[0], filter(lambda x: x[1] == max_score, scores)))
        
        final_hosts = set()
        for service in result:
            for host in hosts:
                if (graph.has_edge(service, host[0])):
                    final_hosts.add(host[0])

        # final hosts -> correlation
        
            
        return final_hosts
                
                    

if __name__ == '__main__':
    # simulate usage from the upper model
    # receives KPI and Trace information as a dataframe in the given window interval

    # load trace data
    # load kpi

    traces = pd.read_csv('data/small_trace.csv').drop(['Unnamed: 0'], axis=1)
    kpis = pd.read_csv('data/small_kpis.csv').drop(['Unnamed: 0'], axis=1)

    # print(traces)

    microRCA = MicroRCA()

    res = microRCA.detect(traces, kpis, visualize=False)

    print(f'Result {res}')