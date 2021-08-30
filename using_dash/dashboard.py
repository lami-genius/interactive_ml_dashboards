import dash # v 1.16.2
import dash_core_components as dcc # v 1.12.1
import dash_bootstrap_components as dbc # v 0.10.3
import dash_html_components as html # v 1.1.1
import pandas as pd
import plotly.express as px # plotly v 4.7.1
import plotly.graph_objects as go
import numpy as np

external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Interactive Model Dashboard', external_stylesheets=[external_stylesheets])

df = pd.read_csv('Data/customer_dataset.csv')
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']
models = ['PCA', 'UMAP', 'AE', 'VAE']
df_average = ###
max_val = ###

app.layout = html.Div([
    html.Div([

        html.Div([
            
            html.Div([
                html.Label('Model selection'),], style={'font-size': '18px'}),
            
            dcc.Dropdown(
                id='crossfilter-model',
                options=[
                    ###
                ],
                ###
                ###

            )], style={'width': '49%', 'display': 'inline-block'}
        ),

        html.Div([
            
            html.Div([
                html.Label('Feature selection'),], style={'font-size': '18px', 'width': '40%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.RadioItems(
                    id='gradient-scheme',
                    options=[
                        ###
                    ],
                    value='Plasma',
                    labelStyle={'float': 'right', 'display': 'inline-block', 'margin-right': 10}
                ),
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
            
            dcc.Dropdown(
                id='crossfilter-feature',
                options=###
                value='None',
                clearable=False
            )], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
        
        )], style={'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px 5px'}
    ),

    html.Div([

        dcc.Graph(
            id=###
            hoverData={'points': [{'customdata': 0}]}
        )

    ], style={'width': '100%', 'height':'90%', 'display': 'inline-block', 'padding': '0 20'}),
    
    html.Div([
        dcc.Graph(id='point-plot'),
    ], style={'display': 'inline-block', 'width': '100%'}),

    ], style={'backgroundColor': 'rgb(17, 17, 17)'},
)


@app.callback(
    ###
)
def update_graph(feature, model, gradient):
    return None


def create_point_plot(df, title):
    return None


@app.callback(
    ###
)
def update_point_plot(hoverData):
    return None


if __name__ == '__main__':
    app.run_server(debug=True)
