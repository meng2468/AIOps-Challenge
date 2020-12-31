import pandas as pd
from sklearn.cluster import Birch
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

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
        traces_df = self.process(traces_df)

        # Parse the traces and kpis
        parsed_traces = self.parse_traces(traces_df)
        
        # FIXME possible case where system doesn't answer for a long time and wasn't called

        #check for anomaly
        # 1 - find outlier in elapsed
        #   1.1 microRCA

        traces = self.get_anomalous_traces(traces_df)
        
        # Hosts + Service
        # Each service connects to all the services it communicates with and all hosts it connects to (no need to differentiate!)
        DG = nx.DiGraph()
        for trace in traces:
            DG = self.trace_graph(parsed_traces[trace], DG)
        
        if visualize:
            print(DG.nodes(data=True), len(DG.nodes()))

            plt.figure(figsize=(9,9))
            #pos = nx.spring_layout(DG)
            # pos = nx.draw_shell(DG)
            # nx.draw(DG, pos, with_labels=True, cmap=plt.get_cmap('jet'), node_size=0, arrows=True)
            nx.draw_shell(DG, with_labels=True)
            # nx.draw_networkx_nodes(DG, pos, nodelist=hosts, node_color="r", node_size=1500)
            # nx.draw_networkx_nodes(DG, pos, nodelist=services, node_color="b", node_size=500)
            # nx.draw_networkx_edges(DG, pos, width=1.0, alpha=0.5)

            labels = nx.get_edge_attributes(DG, 'weight')
            # nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels)
            plt.savefig('output.png')

        # print(f'[DEBUG] Graph is {"connected" if nx.is_weakly_connected(DG) else "not connected"}')
        # Extract anomalous subgraph
        anomaly_DG, anomalous_edges = self.get_anomalous_graph(DG, traces, parsed_traces)

        if nx.is_empty(anomaly_DG):
            raise ValueError('No anomaly detected')

        # Faulty service localization
        # Update weights of anomalous graph
        #           Use cases from the paper
        # Get personalization vector (Transition Probability Matrix)
        # Reverse the service-service edges
        # Apply pagerank
        parsed_kpis = self.parse_kpis(kpis)
        
        result = self.get_fault_service(anomaly_DG, anomalous_edges, traces_df, parsed_kpis)

        return result

    def process(self, traces_df):
        ids = traces_df[traces_df['callType'] == 'CSF']
        
        relationship = {}
        
        def parse(row):
            # parent -> child
            if row['pid'] in ids:
                relationship[row['pid']] = row['cmdb_id']
                

        def apply(row):
            # parent -> new_parent
            if row['callType'] != 'CSF':
                return row
            else:
                if row['id'] in relationship:
                    row['cmdb_id'] = relationship[row['id']]
                return row

        traces_df.apply(parse, axis=1)
        return traces_df.apply(apply, axis=1)

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
        return traces 

    def parse_kpis(self, kpis):
        return dict(tuple(kpis.groupby('cmdb_id')))


    def anomalus_subgraph(self, DirectedGraph, anomalies, ):
        pass

    def trace_graph(self, trace, prev_graph, visualize=False):
        DG = nx.DiGraph(prev_graph)
        
        hosts = trace['cmdb_id'].unique()
        services = trace['serviceName'].unique()

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

    def get_anomalous_graph(self, graph, anomalousids, traces):
        anomalous_nodes = set()
        
        def detect_nodes(row):
            # {k -> v} k = serviceName + callType 
            model = self.detectors[row.loc['serviceName'] + row.loc['callType']]
            prediction = model.predict(np.array(row['elapsedTime']).reshape(-1,1))
            if prediction != model.anomaly_index: # FIXME anomaly_index is inverted. It signals the normal index!!!
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
        # print(f"[DEBUG] Anomalous edges are {anomalous_edges}")
        
        anomalous_nodes = list(anomalous_graph.nodes)
        # print(f"[DEBUG] Anomalous nodes are {anomalous_nodes}")
        for node in anomalous_nodes:
            for n in graph.predecessors(node):
                if n not in anomalous_graph.nodes:
                    anomalous_graph.add_node(n, status='normal', **graph.nodes[n])
            
            for n in graph.successors(node):
                if n not in anomalous_graph.nodes:
                    anomalous_graph.add_node(n, status='normal', **graph.nodes[n])
            
            # anomalous_graph.add_edges_from(list(map(lambda x: x[::-1], graph.in_edges(node))))
            # anomalous_graph.add_edges_from(graph.out_edges(node))
            for n in graph.predecessors(node):
                if n != node:
                    anomalous_graph.add_edge(n, node)
            
            for n in graph.successors(node):
                if n != node:
                    anomalous_graph.add_edge(node, n)
            
        return anomalous_graph, anomalous_edges

    def get_transformation(self, series):
        scaler = StandardScaler()
        val = series.values.reshape(-1,1)
        return pd.Series(scaler.fit_transform(val)[:,0])

    def get_max_correlation(self, times, host_kpi_df):
        """ Returns absolute value of correlation """
        kpi_list = ['container_cpu_used', 'Proc_User_Used_Pct', 'Proc_Used_Pct', 'Sess_Connect', 'On_Off_State', 'tnsping_result_time', 'Sent_queue', 'Received_queue']
        max_corr = 0
        key = None
        correct_times = self.get_transformation(pd.Series(list(times)))

        for k in host_kpi_df['name'].unique():
            if k not in kpi_list:
                continue

            serie = host_kpi_df[host_kpi_df['name'] == k]['value']
            if len(serie.index) == 1 or serie.var() == 0:
                continue
            
            count = round(len(correct_times) / len(serie) + 0.5)
            correct_serie = pd.Series(list(serie))
            correct_serie = self.get_transformation(correct_serie.repeat(count))
            correct_serie = correct_serie.reset_index(drop=True)

            correct_serie = correct_serie[:len(correct_times)] # trim to correct length

            corr = correct_serie.corr(correct_times) # FIXME divide by zero sometimes?

            if math.isnan(corr): # in case kpis don't vary
                print('[ERROR] THIS SHOULD NOT HAPPEN LINE 269')
                corr = 0
            
            corr = abs(corr)
            if corr > max_corr:
                max_corr = corr
                key = k

        return max_corr, key

    def service_to_service_correlation(self, time1, time2):
        time1 = time1.sort_values('startTime')['elapsedTime']
        time2 = time2.sort_values('startTime')['elapsedTime']
        size = min(len(time1),len(time2))
        time1 = self.get_transformation(time1[-size:].reset_index(drop=True))
        time2 = self.get_transformation(time2[-size:].reset_index(drop=True))

        return abs(time1.corr(time2))

    def get_fault_service(self, graph, anomalous_edges, traces, kpis):
        for v in graph.nodes:
            if graph.nodes[v]['status'] != 'anomaly':
                continue
            in_val = 0
            for edge in graph.in_edges(v):
                src, _ = edge
                if edge in anomalous_edges:
                    weight = self.weights_alpha  
                else: 
                    weight = self.service_to_service_correlation(
                        traces[traces['serviceName'] == v],
                        traces[traces['serviceName'] == src]
                    )
                if math.isnan(weight): # in case kpis don't vary
                    print('[ERROR] THIS SHOULD NOT HAPPEN LINE 288')
                    weight = 0

                in_val += weight
                # new_edge = (src, v, {'weight': round(weight,3)})
            
                nx.set_edge_attributes(graph, {(src, v) : {'weight' : weight}})
            
            in_val /= graph.in_degree(v)

            for edge in graph.out_edges(v):
                _, dst = edge
                if (edge in anomalous_edges):
                    continue
                if graph.nodes[dst]['type'] == 'service':
                    val = self.service_to_service_correlation(
                        traces[traces['serviceName'] == v],
                        traces[traces['serviceName'] == dst]
                    )
                    if math.isnan(val): # in case kpis don't vary
                        print('[ERROR] THIS SHOULD NOT HAPPEN LINE 305')
                        val = 0
                else: # host
                    kpi = kpis[dst]

                    times = traces[traces['serviceName'] == v]['elapsedTime']
                    max_corr, _ = self.get_max_correlation(times, kpi)
                    val = in_val * max_corr
                nx.set_edge_attributes(graph, {(v, dst) : {'weight' : val}})
        

        personalization = {}
        for v in graph.nodes:
            if graph.nodes[v]['status'] != 'anomaly':
                continue    
            
            #get avg weight
            vals = [graph.get_edge_data(*edge)['weight'] for edge in graph.out_edges(v)] #all 
            
            avg = sum(vals) / len(vals)


            # get max correlation value
            max_corr = 0

            times = traces[traces['serviceName'] == v]['elapsedTime']
            most_prob_host = None
            keys = filter(lambda x: graph.nodes[x[1]]['type']=='host', graph.out_edges(v))
            kpi_name = None
            for key in keys:
                kpi_df = kpis[key[1]]
                val, kpi = self.get_max_correlation(times, kpi_df)
                if val > max_corr:
                    most_prob_host = key[1]
                    max_corr = val
                    kpi_name = kpi
            
            val = avg * max_corr
            personalization[v] = val / graph.degree(v) # why do they do this in the original code?
            graph.nodes[v]['most_probable_host'] = most_prob_host
            graph.nodes[v]['kpi_name'] = kpi_name

        # print(f'[DEBUG] Personalization vector: {personalization}')
        anomalous = any(map(lambda x: x > 0, personalization.values()))
        # if not anomalous:
        #     raise ValueError('No anomaly could be found.')
        if not anomalous:
            N = len(personalization)
            personalization = dict(map(lambda x: (x, 1/N), personalization))

        reversed_graph = graph.reverse(copy=True)

        # transform hosts to have only in_edges
        for node in reversed_graph.nodes:
            if reversed_graph.nodes[node]['type'] != 'host':
                continue

            edges = list(reversed_graph.out_edges(node))
            for edge in edges:
                src, dst = edge
                data = reversed_graph.get_edge_data(src, dst)
                reversed_graph.add_edge(dst, src, **data)
                reversed_graph.remove_edge(src,dst)
                

        scores = nx.pagerank(
            reversed_graph, 
            alpha=self.page_rank_alpha,
            personalization=personalization,
            max_iter=self.page_rank_max_iter
        )
        scores = sorted(filter(lambda x: x[1] > 0, scores.items()), key=lambda x: x[1], reverse=True)
        # print(f'[DEBUG] Scores {scores}')
        hosts = list(filter(lambda x: x[1]['type'] == 'host', reversed_graph.nodes(data=True)))
       
        # p(h | s) = p(s)*p(s | h) / p(h)

        total_sum = sum(map(lambda x: x[1], scores))
        percentual_service = list(map(lambda x: (x[0], x[1] / total_sum), scores))

        total_weights = 0
        for host in hosts:
            edges = reversed_graph.in_edges(host[0])
            host_sum = sum([reversed_graph.get_edge_data(u,v)['weight'] for u,v in edges])
            total_weights += host_sum
            reversed_graph.nodes[host[0]]['host_sum'] = host_sum

        
        final_result = []
        for host in hosts:
            for service in scores:
                if not reversed_graph.has_edge(service[0], host[0]):
                    continue
                weight = reversed_graph.get_edge_data(service[0], host[0])['weight']

                if weight == 0:
                    continue
                
                ph = reversed_graph.nodes[host[0]]['host_sum'] / total_weights
                ps = service[1] / total_sum
                
                times = traces[traces['serviceName'] == service[0]]['elapsedTime']
                final_result.append((host[0], service[0], self.get_max_correlation(times, kpis[host[0]])[1], ps * weight / ph))
        

        total_val = sum(map(lambda x: x[3], final_result))
        percentual_results = list(map(lambda x: [*x[0:3], x[3] / total_val], final_result))
        sorted_results = sorted(percentual_results, key=lambda x: -x[3]) #hosts_info #sorted(final_result, key=lambda x: x[2])
        formatted_answer = []
        cumsum = 0
        for res in sorted_results:
            formatted_answer.append([res[0],res[2]])
            cumsum += res[3]
            if cumsum > 0.8:
                break

        return formatted_answer
                
                    

if __name__ == '__main__':
    # simulate usage from the upper model
    # receives KPI and Trace information as a dataframe in the given window interval

    # load trace data
    # load kpi

    traces = pd.read_csv('/mnt/c/Users/tiago/Documents/Uni/anm/anm-project/data/labeled_data/AIOps挑战赛数据/2020_04_11/2020_04_11/test1/traces.csv').drop(['Unnamed: 0'], axis=1)
    kpis = pd.read_csv('/mnt/c/Users/tiago/Documents/Uni/anm/anm-project/data/labeled_data/AIOps挑战赛数据/2020_04_11/2020_04_11/test1/kpis.csv').drop(['Unnamed: 0'], axis=1)


    microRCA = MicroRCA()

    res = microRCA.detect(traces, kpis, visualize=False)

    print(f'Result {res}')