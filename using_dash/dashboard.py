import dash  # v 1.16.2
import dash_core_components as dcc  # v 1.12.1
import dash_bootstrap_components as dbc  # v 0.10.3
import dash_html_components as html  # v 1.1.1
import pandas as pd
import plotly.express as px  # plotly v 4.7.1
import plotly.graph_objects as go
import numpy as np
import math
from dash.dependencies import Output, Input

external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Interactive Model Dashboard',
                external_stylesheets=[external_stylesheets])

df = pd.read_csv('data/customer_dataset.csv')
features = ['Channel', 'Region', 'Fresh', 'Milk', 'Grocery',
            'Frozen', 'Detergents_Paper', 'Delicatessen', 'Total_Spend']
models = ['PCA', 'UMAP', 'AE', 'VAE']
user_view_models = ['Principal Component Analysis',
                    'Uniform Manifold Approximation and Projection',
                    'Autoencoder', 'Variational Autoencoder']
df_average = df[features].mean()
max_val = max(df[features].max())
min_size = 20000

app.layout = html.Div([
    html.Div([
        html.Div([

            html.Div([
                html.Label('Model selection'), ], style={'font-size': '18px'}),

            dcc.Dropdown(
                id='crossfilter-model',
                options=[
                    # Dictionary for models - value and label
                    {'label': 'Principal Component Analysis', 'value': 'PCA'},
                    {'label': 'Uniform Manifold Approximation and Projection',
                        'value': 'UMAP'},
                    {'label': 'Autoencoder', 'value': 'AE'},
                    {'label': 'Variational Autoencoder', 'value': 'VAE'}, ],
                value='PCA',
                clearable=False,
            )], style={'width': '49%', 'display': 'inline-block'}
        ),

        html.Div([

            html.Div([
                html.Label('Feature selection'), ], style={'font-size': '18px', 'width': '40%', 'display': 'inline-block'}),

            html.Div([
                dcc.RadioItems(
                    id='gradient-scheme',
                    options=[
                       {'label': 'Orange to Red', 'value': 'OrRd'},
                       {'label': 'Viridis', 'value': 'Viridis'},
                       {'label': 'Plasma', 'value': 'Plasma'},
                    ],
                    value='Plasma',
                    labelStyle={'float': 'right',
                                'display': 'inline-block', 'margin-right': 10, 'color': 'white'}
                ),
            ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),

            dcc.Dropdown(
                id='crossfilter-feature',
                options=[{'label': feature, 'value': feature}
                         for feature in ['None'] + features],
                value='None',
                clearable=False
            )], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}

        )], style={'backgroundColor': 'rgb(17, 17, 17)', 'padding': '10px 5px'}
    ),

    html.Div([

        dcc.Graph(
            id='scatter-plot',
            hoverData={'points': [{'customdata': 0}]}
        )

    ], style={'width': '100%', 'height': '90%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        dcc.Graph(id='point-plot'),
    ], style={'display': 'inline-block', 'width': '100%'}),

], style={'backgroundColor': 'rgb(17, 17, 17)'},
)


@app.callback(
    Output('scatter-plot', 'figure'),
    [
        Input('crossfilter-feature', 'value'),
        Input('crossfilter-model', 'value'),
        Input('gradient-scheme', 'value'),
    ]
)
def update_graph(feature, model, gradient):

    if feature == 'None':
        colors = None
        sizes = None
        hover_names = [f'Customer {ix}' for ix in df.index]
    elif feature in ['Region', 'Channel']:
        colors = df[feature].astype(str)
        sizes = None
        hover_names = [f'Customer {ix}' for ix in df.index]
    else:
        colors = df[feature]
        sizes = df[feature] + min_size
        hover_names = []
        for ix, val in zip(df.index.values, df[feature].values):
            hover_names.append(f'Customer {ix}<br>{feature} value of {val}')

    fig = px.scatter(
        df,
        x=df[f'{model.lower()}_x'],
        y=df[f'{model.lower()}_y'],
        opacity=0.8,
        template='plotly_dark',
        color_continuous_scale=gradient,
        hover_name=hover_names,
        color=colors,
        size=sizes,
    )

    # Update single data
    fig.update_traces(
        customdata=df.index
    )

    fig.update_layout(
        coloraxis_showscale=False,
        hovermode='closest',
        height=440,
        margin={'r': 10, 'l': 10, 't': 10, 'b': 10},

    )

    # Remove tick labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def create_point_plot(df, title):
    fig = go.Figure(
        data=[
            go.Bar(name='Average', x=features,
                   y=df_average.values, marker_color="#c178f6"),
            go.Bar(name=title, x=features, y=df.values, marker_color="#8aefbd"),
        ]
    )

    fig.update_layout(
        barmode='group',
        height=250,
        margin={'r': 10, 'l': 10, 't': 10, 'b': 10},
        template='plotly_dark'

    )

    fig.update_xaxes(
        showgrid=False)

    fig.update_yaxes(
        type='log',
        range=[0, math.ceil(np.log10(max_val))],
    )

    return fig


@app.callback(
    Output('point-plot', 'figure'),
    [Input('scatter-plot', 'hoverData')],
)
def update_point_plot(hoverData):
    index = hoverData['points'][0]['customdata']
    title = f'Customer {index}'
    return create_point_plot(df[features].iloc[index], title)


if __name__ == '__main__':
    app.run_server(debug=False)
