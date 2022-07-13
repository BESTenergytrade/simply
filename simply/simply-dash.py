from dash import Dash, dcc, Output, Input, html
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
import networkx as nx
import pandas as pd
import json

from market_fair import BestMarket
from actor import Order
from power_network import PowerNetwork

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

agents = {
    0: {'cluster': 'cluster 0', 'edges': [1, 4]},
    1: {'cluster': 'cluster 1', 'edges': [0, 2]},
    2: {'cluster': 'cluster 1', 'edges': [1]},
    3: {'cluster': 'cluster 1', 'edges': [1]},
    4: {'cluster': 'cluster 0', 'edges': [0]}
}

clusters = ['cluster 0', 'cluster 1']
edges = [(0, 1, {"weight": 1}), (1, 2), (1, 3), (0, 4)]

nw = nx.Graph()
nw.add_edges_from(edges)
pn = PowerNetwork("", nw, weight_factor=1)
m = BestMarket(0, pn)

# add bids
m.accept_order(Order(-1, 0, 1, 1, 0.1, 10))
m.accept_order(Order(-1, 0, 1, 1, 0.1, 7))
m.accept_order(Order(-1, 0, 0, 0, 0.1, 10))
# add asks
m.accept_order(Order(1, 0, 3, 1, 0.1, 6))
m.accept_order(Order(1, 0, 3, 1, 0.1, 4))
orders_df = pd.DataFrame(m.orders)

matches = m.match()


# Preprocess match data
matches_df = pd.DataFrame(matches)

app = Dash(external_stylesheets=[dbc.themes.FLATLY])

title = html.H1(children='Simply Matching Algorithms')
subtitle = html.Div(children='Interactive web application for simply matching algorithms')

selected_matches_table = html.Div(id="table-container", children='')

dropdown = dbc.DropdownMenu(id='dropdown',
                            label="Algorithm",
                            children=[
                                dbc.DropdownMenuItem("Pay-as-Bid", id='Market', n_clicks=0),
                                dbc.DropdownMenuItem("Two-sided Pay-as-Clear", id='TwoSidedPayAsClear', n_clicks=0),
                                dbc.DropdownMenuItem("BEST Matching", id='BestMarket', n_clicks=0),
                            ],
                            )

orders_title = html.H3(children='Orders:')
orders_table =  dbc.Table.from_dataframe(orders_df,
                                             id='orders_table',
                                             striped=False,
                                             bordered=True,
                                             hover=True)

selected_algorithm = html.Div(id='selected-algorithm')

match_button = html.Div(
    [
        dbc.Button("Match", color="primary"), ],
    className="d-grid gap-2 col-6 mx-auto",
)

all_matches_title = html.H3(children='Total matches:')
all_matches_table = dbc.Table.from_dataframe(matches_df,
                                             id='matches_table',
                                             striped=False,
                                             bordered=True,
                                             hover=True)

cyto = html.Div([
    cyto.Cytoscape(
        autoungrabify=True,
        id='cytoscape-compound',
        layout={'name': 'cose'},
        boxSelectionEnabled=True,

        style={'width': '100%', 'height': '500px'},
        elements=
        # Parent nodes
        [{'data': {'id': str(val['cluster']), 'label': str(val['cluster'])}} for x,
                                                                                 val in
         agents.items()] +
        # Children nodes
        [{'data': {'id': str(x), 'label': str(x), 'parent': val['cluster']}} for x,
                                                                                 val in
         agents.items()] +
        [
            # Edge elements
            {'data': {'source': str(edge[0]), 'target': str(edge[1])}} for edge in edges
        ]
    )
])

app.layout = dbc.Container([title, subtitle, cyto, orders_title, orders_table, selected_algorithm,
                            dropdown,
                            match_button,
                            selected_matches_table,
                            all_matches_title,
                            all_matches_table])


@app.callback(Output("table-container", "children"),
              Input('cytoscape-compound', 'tapNodeData'))
def update_dataframe(data):
    # return json.dumps(data, indent=2)
    bid_actors = matches_df[matches_df['bid_actor'] == int(data['id'])]
    ask_actors = matches_df[matches_df['ask_actor'] == int(data['id'])]
    display = pd.concat([bid_actors, ask_actors])
    if len(display) > 0:
        selected_matches_title = html.H3(children='Selected matches:')
        return [selected_matches_title, dbc.Table.from_dataframe(display, id='matches_table',
                                                                 striped=False, bordered=True,
                                                                 hover=True, color='blue')]


@app.callback(Output("selected-algorithm", "children"),
              [Input('Market', 'id'), Input('TwoSidedPayAsClear', 'id'), Input('BestMarket', 'id')])
def update_algorithm(n):
    print(n)

if __name__ == '__main__':
    app.run_server(debug=True)
