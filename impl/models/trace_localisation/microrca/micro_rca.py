# import requests
import argparse
import csv
import datetime
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import Birch


class MicroRCA():
    """
    Class implementing the MicroRCA algorithm as described in:

    Li Wu, Johan Tordsson, Erik Elmroth, Odej Kao.
    MicroRCA: Root Cause Localization of Performance Issues in Microservices
    IEEE/IFIP Network Operations and Management Symposium (NOMS), 20-24 April 2020, Budapest, Hungary
    
    """
    def __init__(self, 
        data_dir,
        faults_type, 
        targets, 
        smoothing_window=12, 
        min_periods=1, 
        branching_factor=50, 
        ad_threshold=0.045, 
        weights_alpha=0.55, 
        page_rank_alpha=0.85, 
        page_rank_max_iter=10000,
        debug=False
    ):
        """
        - Input parameters
        faults_type (list): KPIs to analyze for anomalies
        targets (list): List of services
        smoothing_window (int): Parameter for the windowing in BIRCH preprocessing (default = 12)
        min_periods (int): Minute period for the windowing in BIRCH preprocessing (default = 1)
        branching_factor (int): Parameter for the BIRCH algorithm (default = 50)
        ad_threshold (float): Threshold for the anomaly detection step of BIRCH algorithm
        alpha (float): Confidence parameter for the weights of edges between 2 abnormal nodes

        - Example:
        faults_type = ['svc_latency', 'service_cpu', 'service_memory']
        targets = ['front-end', 'catalogue', 'orders', 'user', 'carts', 'payment', 'shipping']

        mrca = MicroRCA(faults_type, targets)

        """
        self.data_dir = data_dir
        self.faults_type = faults_type
        self.targets = targets
        
        self.smoothing_window = smoothing_window
        self.min_periods = min_periods

        self.branching_factor = branching_factor 
        self.ad_threshold = ad_threshold
        
        self.weights_alpha = weights_alpha

        self.page_rank_alpha = page_rank_alpha
        self.page_rank_max_iter = page_rank_max_iter

        self.debug = debug


    def detect(self, faults_type, target):
        faults_name = os.path.join(self.data_dir, faults_type + '_' + target)

        # Setp 1: get the response times 
        latency_df = self.rt_invocations(faults_name)

        # Step 2: Get anomalies from the clustering with BIRCH
        anomaly_nodes = []
        anomalies = self.birch_ad_with_smoothing(latency_df)
        for anomaly in anomalies:
            edge = anomaly.split('_')
            anomaly_nodes.append(edge[1])
        anomaly_nodes = set(anomaly_nodes)

        # Step 3: Build the graph from the node lists
        DG = self.build_graph(faults_name, visualize=self.debug)

        # Step 4: Extract the anomalus subgraph
        # anomaly_graph, personalization = self.anomaly_subgraph(DG, anomaly_nodes, latency_df, faults_name, self.weights_alpha)
        # Original line had "anomalies", shouldnt it be "anomaly_nodes" set?
        anomaly_graph, personalization = self.anomaly_subgraph(DG, anomalies, latency_df, faults_name, self.weights_alpha) 
      
        # Step 5: Calculate the anomaly score by running PageRank
        anomaly_scores = self.anomaly_score(anomaly_graph, personalization)
        return anomaly_scores

    def rt_invocations(self, faults_name):
        latency_filename = faults_name + '_latency_source_50.csv'  # inbound
        latency_df_source = pd.read_csv(latency_filename) 
        # latency_df_source['unknown_front-end'] = 0
        
        latency_filename = faults_name + '_latency_destination_50.csv' # outbound
        latency_df_destination = pd.read_csv(latency_filename) 
        
        latency_df = latency_df_destination.add(latency_df_source)    

        #################################################
        # FIXME: The data provided in the original repo is messed up, remove this filtering with our data
        latency_df = latency_df[['orders_user']]
        #################################################

        return latency_df

    
    def birch_ad_with_smoothing(self, latency_df):
        if(latency_df is None):
            latency_df = self.rt_invocations()

        anomalies = []
        for svc, latency in latency_df.iteritems():
            # No anomaly detection in db  # TODO maybe change this to detect anomalies in all services
            if svc != 'timestamp' and 'Unnamed' not in svc and 'rabbitmq' not in svc and 'db' not in svc:
                latency = latency.rolling(window=self.smoothing_window, min_periods=1).mean()
                x = np.array(latency)
                x = np.where(np.isnan(x), 0, x)
                normalized_x = preprocessing.normalize([x])

                X = normalized_x.reshape(-1,1)

                brc = Birch(
                    branching_factor=self.branching_factor, 
                    n_clusters=None, 
                    threshold=self.ad_threshold, 
                    compute_labels=True
                )
                brc.fit(X)
                brc.predict(X)

                labels = brc.labels_
    #            centroids = brc.subcluster_centers_
                n_clusters = np.unique(labels).size
                if n_clusters > 1:
                    anomalies.append(svc)
        return anomalies


    def build_graph(self, faults_name, visualize=False):
        # build the attributed graph 
        # input: prefix of the file
        # output: attributed graph

        filename = faults_name + '_mpg.csv'
        df = pd.read_csv(filename)
        
        DG = nx.DiGraph()
        for index, row in df.iterrows():
            source = row['source']
            destination = row['destination']
            # if 'rabbitmq' not in source and 'rabbitmq' not in destination and 'db' not in destination and 'db' not in source:
            DG.add_edge(source, destination)

        for node in DG.nodes():
            if 'kubernetes' in node: # TODO adapt to our case for hosts / services
                DG.nodes[node]['type'] = 'host'
            else:
                DG.nodes[node]['type'] = 'service'
       
        if visualize:
            print(DG.nodes(data=True))
                        
            plt.figure(figsize=(9,9))
            nx.draw(DG, with_labels=True, font_weight='bold')
            pos = nx.spring_layout(DG)
            nx.draw(DG, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True)
            labels = nx.get_edge_attributes(DG, 'weight')
            nx.draw_networkx_edge_labels(DG, pos, edge_labels=labels)
            plt.show()
                    
        return DG 


    def svc_personalization(svc, anomaly_graph, baseline_df, faults_name):
        filename = faults_name + '_' + svc + '.csv'
        df = pd.read_csv(filename)
        ctn_cols = ['ctn_cpu', 'ctn_network', 'ctn_memory']
        max_corr = 0.01
        metric = 'ctn_cpu'
        for col in ctn_cols:
            temp = abs(baseline_df[svc].corr(df[col]))     
            if temp > max_corr:
                max_corr = temp
                metric = col

        edges_weight_avg = 0.0
        num = 0
        for u, v, data in anomaly_graph.in_edges(svc, data=True):
            num = num + 1
            edges_weight_avg = edges_weight_avg + data['weight']

        for u, v, data in anomaly_graph.out_edges(svc, data=True):
            if anomaly_graph.nodes[v]['type'] == 'service':
                num = num + 1
                edges_weight_avg = edges_weight_avg + data['weight']

        edges_weight_avg  = edges_weight_avg / num

        personalization = edges_weight_avg * max_corr

        return personalization, metric    

    
    def node_weight(self, svc, anomaly_graph, baseline_df, faults_name):
        #Get the average weight of the in_edges
        in_edges_weight_avg = 0.0
        num = 0
        for u, v, data in anomaly_graph.in_edges(svc, data=True):
            # TODO: cant we just check the degrees here?
            num = num + 1
            in_edges_weight_avg = in_edges_weight_avg + data['weight']
        if num > 0:
            in_edges_weight_avg  = in_edges_weight_avg / num 

        filename = faults_name + '_' + svc + '.csv'
        df = pd.read_csv(filename)
        node_cols = ['node_cpu', 'node_network', 'node_memory']
        max_corr = 0.01
        metric = 'node_cpu'
        for col in node_cols:
            temp = abs(baseline_df[svc].corr(df[col]))
            if temp > max_corr:
                max_corr = temp
                metric = col
        data = in_edges_weight_avg * max_corr
        return data, metric


    def anomaly_subgraph(self, DG, anomalies, latency_df, faults_name, visualize=False):
        # Get the anomalous subgraph
        # input: 
        #   DG: attributed graph
        #   anomlies: anoamlous service invocations
        #   latency_df: service invocations from data collection
        #   agg_latency_dff: aggregated service invocation
        #   faults_name: prefix of csv file
        #   alpha: weight of the anomalous edge
        #
        # output:
        #   anomalous graph
        #   personalization

        if DG == []:
            DG = self.build_graph(faults_name)

        # Get reported anomalous nodes

        # Added the anomalous_ prefix to variables 
        anomalous_edges = []
        anomalous_nodes = []
        baseline_df = pd.DataFrame()
        edge_df = {}
        for anomaly in anomalies:
            edge = anomaly.split('_')
            anomalous_edges.append(tuple(edge))
            # nodes.append(edge[0])
            svc = edge[1]
            anomalous_nodes.append(svc)
            baseline_df[svc] = latency_df[anomaly]
            edge_df[svc] = anomaly

        # Set contains the anomalous nodes (edge[1] for all anomalous egdes)
        anomalous_nodes = set(anomalous_nodes)

        # Find the anomalous nodes in the graph
        personalization = {}
        for node in DG.nodes():
            if node in anomalous_nodes:
                personalization[node] = 0

        # Get the subgraph of anomaly
        anomaly_graph = nx.DiGraph()
        for node in anomalous_nodes:
            # For later: do we merge the two blocks into a single function that processes in/out bound egdes?

            # Process inbound edges
            for u, v, data in DG.in_edges(node, data=True):
                edge = (u,v)
                if edge in anomalous_edges:
                    data = self.weights_alpha
                else:
                    normal_edge = u + '_' + v
                    data = baseline_df[v].corr(latency_df[normal_edge])

                data = round(data, 3)
                anomaly_graph.add_edge(u,v, weight=data)
                anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
                anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

            # Set personalization with container resource usage
            # Process outbound edges
            for u, v, data in DG.out_edges(node, data=True):
                edge = (u,v)
                if edge in anomalous_edges:
                    data = self.weights_alpha
                else:
                    if DG.nodes[v]['type'] == 'host':
                        data, col = self.node_weight(u, anomaly_graph, baseline_df, faults_name)
                    else:
                        normal_edge = u + '_' + v
                        data = baseline_df[u].corr(latency_df[normal_edge])

                data = round(data, 3)
                anomaly_graph.add_edge(u,v, weight=data)
                anomaly_graph.nodes[u]['type'] = DG.nodes[u]['type']
                anomaly_graph.nodes[v]['type'] = DG.nodes[v]['type']

        for node in anomalous_nodes:
            max_corr, col = self.svc_personalization(node, anomaly_graph, baseline_df, faults_name)
            personalization[node] = max_corr / anomaly_graph.degree(node)

        # Edges are REVERSED here!
        anomaly_graph = anomaly_graph.reverse(copy=True)

        # These are also anomalous edges
        edges = list(anomaly_graph.edges(data=True))

        # Update weights for the hosts. Notice that the edge is created in REVERSE again v->u
        for u, v, d in edges:
            if anomaly_graph.nodes[node]['type'] == 'host': # TODO check this "node" indexing 
                anomaly_graph.remove_edge(u,v)
                anomaly_graph.add_edge(v,u,weight=d['weight'])

        # if visualize:
        #     plt.figure(figsize=(9,9))
        #     pos = nx.spring_layout(anomaly_graph)
        #     nx.draw(anomaly_graph, pos, with_labels=True, cmap = plt.get_cmap('jet'), node_size=1500, arrows=True, )
        #     labels = nx.get_edge_attributes(anomaly_graph,'weight')
        #     nx.draw_networkx_edge_labels(anomaly_graph,pos,edge_labels=labels)
        #     plt.show()
        #     print('Personalization:', personalization)

        return anomaly_graph, personalization


    def anomaly_score(self, anomaly_graph, personalization):
        scores = nx.pagerank(
            anomaly_graph, 
            alpha=self.page_rank_alpha, 
            personalization=personalization, 
            max_iter=self.page_rank_max_iter
        )
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return scores


def print_rank(anomaly_score, target):
    num = 10
    for idx, anomaly_target in enumerate(anomaly_score):
        if target in anomaly_target:
            num = idx + 1
            continue
    print(target, ' Top K: ', num)
    return num


if __name__ == "__main__":

    data_dir = os.path.join('data')
    faults_type = ['service_cpu']
    targets = ['orders']

    mrca = MicroRCA(
        data_dir=data_dir, 
        faults_type=faults_type,
        targets=targets,
        debug=False
    )

    anomaly_score = mrca.detect(
        faults_type=faults_type[0],
        target=targets[0]
    )

    print_rank(
        anomaly_score, 
        target=targets[0]
    )