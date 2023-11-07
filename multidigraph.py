import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from simply import config as cfg
from netgraph import Graph


def main(project_dir, data_dir):
    print(f'project_dir: {project_dir}')
    # File path for the matches csv
    project_dir = Path(project_dir)
    matches_fp = project_dir + 'market_results/matches.csv'  #'results_inputs/matches_dummy_2.csv'

    # Saves matches csv as a dataframe
    matches_df = pd.read_csv(matches_fp)

   # matches_df['time'] = pd.to_datetime(matches_df['time'])

    #matches_df_slice = matches_df.query('"2021-01-01 00:00:00" <= time <= "2021-01-01 14:00:00"')

    cluster_energy_df = matches_df_slice.groupby(['bid_cluster', 'ask_cluster'])['energy'].sum().reset_index()
    # filters out edges with same source and target (where cluster buys/sells energy internally)
    cluster_energy_df = cluster_energy_df[cluster_energy_df['bid_cluster'] != cluster_energy_df['ask_cluster']]

    hh_energy_df = matches_df_slice.groupby(['bid_actor', 'ask_actor', 'bid_cluster', 'ask_cluster'])['energy'].sum().reset_index()

    # get unique cluster values and sort them
    cluster_values = sorted(set(cluster_energy_df['bid_cluster']) | set(cluster_energy_df['ask_cluster']))
    # get unique household values and sort them
    hh_values = sorted(set(hh_energy_df['bid_actor']) | set(hh_energy_df['ask_actor']))

    # generate cluster names dictionary
    cluster_names = {}
    for i, val in enumerate(cluster_values):
        cluster_names[val] = 'c{}'.format(i + 1)

    # generate household names dictionary
    hh_names = {}
    for i, val in enumerate(hh_values):
        hh_names[val] = 'h{}'.format(i + 1)

    # rename households and clusters in DataFrames
    cluster_energy_df['bid_cluster'] = cluster_energy_df['bid_cluster'].apply(lambda x: cluster_names[x])
    cluster_energy_df['ask_cluster'] = cluster_energy_df['ask_cluster'].apply(lambda x: cluster_names[x])
    hh_energy_df['bid_actor'] = hh_energy_df['bid_actor'].apply(lambda x: hh_names[x])
    hh_energy_df['ask_actor'] = hh_energy_df['ask_actor'].apply(lambda x: hh_names[x])
    hh_energy_df['bid_cluster'] = hh_energy_df['bid_cluster'].apply(lambda x: cluster_names[x])
    hh_energy_df['ask_cluster'] = hh_energy_df['ask_cluster'].apply(lambda x: cluster_names[x])

    bid_cluster_groups = hh_energy_df.groupby('bid_cluster')
    ask_cluster_groups = hh_energy_df.groupby('ask_cluster')

    hh_energy_df_sc = hh_energy_df[hh_energy_df['bid_cluster'] == hh_energy_df['ask_cluster']]
    hh_energy_df_sc = hh_energy_df_sc.groupby(['bid_actor', 'ask_actor', 'bid_cluster', 'ask_cluster'])['energy'].sum().reset_index()
    same_cluster_bids = hh_energy_df_sc.groupby('bid_cluster')
    same_cluster_asks = hh_energy_df_sc.groupby('ask_cluster')

    # Creates empty dict for the total energy sold from each household to different clusters
    # e.g. (0, 2): 40 means household 0 (belonging to cluster 2) sells 40 kWh electricity to other
    # clusters (exact clusters not specified)
    hh_energy_sold_dc_dict = {}
    # Creates empty dict for the total energy bought by each household from different clusters
    # e.g. (4, 1): 50 means household 4 (belonging to cluster 1) buys 50 kWh electricity from other
    # clusters (exact clusters not specified)
    hh_energy_bought_dc_dict = {}
    hh_energy_tf_sc_dict = {}

    # iterate over each group
    for cluster, group in bid_cluster_groups:
        # iterate over each bid_actor in the group
        for actor in group['bid_actor'].unique():
            # filter the rows where bid_actor matches and bid_cluster is not equal to ask_cluster
            filtered_rows_dc = group[(group['bid_actor'] == actor) & (group['bid_cluster'] != group['ask_cluster'])]
            # calculate the total energy transferred
            total_energy_dc = filtered_rows_dc['energy'].sum()
            # add the result to the dictionary
            hh_energy_sold_dc_dict[(actor, cluster)] = total_energy_dc

    for cluster, group in ask_cluster_groups:
        for actor in group['ask_actor'].unique():
            # filter the rows where ask_actor matches and bid_cluster is not equal to ask_cluster
            filtered_rows = group[(group['ask_actor'] == actor) & (group['bid_cluster'] != group['ask_cluster'])]
            # calculate the total energy transferred
            total_energy = filtered_rows['energy'].sum()
            # add the result to the dictionary
            hh_energy_bought_dc_dict[(actor, cluster)] = total_energy

    # iterate over each group
    # ToDo: need to check if this works when there are more than one internal sales in a cluster
    for cluster, group in same_cluster_bids:
        for cluster_2, group_2 in same_cluster_asks:
            # iterate over each ask_actor in the group
            for bid_actor in group['bid_actor'].unique():
                for ask_actor in group['ask_actor'].unique():
                    # calculate the total energy transferred
                    total_energy = hh_energy_df_sc['energy'].sum()
                    # add the result to the dictionary
                    hh_energy_tf_sc_dict[(bid_actor, ask_actor)] = total_energy

    # Creates a multi-directional graph in networkx
    G = nx.MultiDiGraph()
    G.add_edges_from(cluster_energy_df[['bid_cluster', 'ask_cluster']].values, weight=1)

    # add edge attribute for energy
    edge_attrs = {(u, v, k): {'energy': d} for u, v, k, d in G.edges(keys=True, data='energy')}
    nx.set_edge_attributes(G, edge_attrs)

    # iterate through the dictionary and add the edges
    for (src, target), edge in hh_energy_sold_dc_dict.items():
        G.add_edge(src, target, weight=1000)

    # iterate through the dictionary and add the edges
    for (src, target), edge in hh_energy_bought_dc_dict.items():
        G.add_edge(src, target, weight=1000)

    # iterate through the dictionary and add the edges
    for (src, target), edge in hh_energy_tf_sc_dict.items():
        G.add_edge(src, target, weight=1000)

    # iterate through the nodes and set their color and size attributes
    node_color = []
    node_size = []
    for node in G.nodes():
        if node.startswith('c'):
            node_color.append('lightblue')  # set color to red for cluster nodes
            node_size.append(700)  # set size to 200 for cluster nodes
        elif node.startswith('h'):
            node_color.append('pink')  # set color to blue for household nodes
            node_size.append(500)

    widths = {}
    for i, (u, v, w) in enumerate(zip(cluster_energy_df['bid_cluster'], cluster_energy_df['ask_cluster'], cluster_energy_df['energy'])):
        widths[(u, v, i)] = w * 0.02 # set the width based on the energy value
    nx.set_edge_attributes(G, widths, 'width')

    # draw the graph
    pos = nx.spring_layout(G, k=7, seed=1)
    #pos = nx.nx_pydot.pydot_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(G, pos, edge_color='grey', width=list(widths.values()), connectionstyle='arc3, rad = 0.1', arrowstyle='simple,head_length=1.5,head_width=1.2,tail_width=0.2')
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=12, font_weight='bold')
    plt.axis('off')

    plt.show()

    print('done')
    return

