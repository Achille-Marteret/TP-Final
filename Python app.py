# %%
## -------------------------------------------- Achille MARTERET -------------------------------------------- ##
## -------------------------------------------- 2025-2026 - M1 ECAP -------------------------------------------- ##


import dash
from dash import html, dcc, callback, Output, Input, dash_table
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from calendar import month_abbr, month_name

# ======= Initialisation de l'application ======= #

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

#================= Traitement des données ================#
df = pd.read_csv("data.csv", index_col=0)
df = df[['CustomerID', 'Gender', 'Location', 'Product_Category', 'Quantity', 'Avg_Price', 'Transaction_Date', 'Month', 'Discount_pct']]

df['CustomerID'] = df['CustomerID'].fillna(0).astype(int)
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
df['Location'] = df['Location'].apply(lambda x: str(x).title())

df['Total_price'] = df['Quantity'] * df['Avg_Price'] * (1 - (df['Discount_pct'] / 100)).round(3)

df['Location'] = df['Location'].replace(['', 'Nan', 'nan'], np.nan) 
df = df.dropna(subset=['Location'])  # Supprime toutes les lignes où Location est NaN


# ============ Statistiques =============== #

def calculer_chiffre_affaire(data):
    return data['Total_price'].sum()

def frequence_meilleure_vente(data, top=10, ascending=False):
    resultat = pd.crosstab(
        [data['Gender'], data['Product_Category']], 
        'Total vente', 
        values=data['Total_price'], 
        aggfunc= lambda x : len(x), 
        rownames=['Sexe', 'Categorie du produit'],
        colnames=['']
    ).reset_index().groupby(
        ['Sexe'], as_index=False, group_keys=True
    ).apply(
        lambda x: x.sort_values('Total vente', ascending=ascending).iloc[:top, :]
    ).reset_index(drop=True).set_index(['Sexe', 'Categorie du produit'])

    return resultat

def indicateur_du_mois(data, current_month = 12, freq=True, abbr=False): 
    previous_month = current_month - 1 if current_month > 1 else 12
    if freq : 
        resultat = data['Month'][(data['Month'] == current_month) | (data['Month'] == previous_month)].value_counts()
        # sort by index
        resultat = resultat.sort_index()
        resultat.index = [(month_abbr[i] if abbr else month_name[i]) for i in resultat.index]
        return resultat
    else:
        resultat = data[(data['Month'] == current_month) | (data['Month'] == previous_month)].groupby('Month').apply(calculer_chiffre_affaire)
        resultat.index = [(month_abbr[i] if abbr else month_name[i]) for i in resultat.index]
        return resultat


# Barplot
def barplot_top_10_ventes(data) :
    df_plot = frequence_meilleure_vente(data, ascending=True)
    graph = px.bar(
        df_plot,
        x='Total vente', 
        y=df_plot.index.get_level_values(1),
        color=df_plot.index.get_level_values(0), 
        barmode='group',
        title="Frequence des 10 meilleures ventes",
        labels={"x": "Fréquence", "y": "Categorie du produit", "color": "Sexe"},
        width=680, height=600
    ).update_layout(
        margin = dict(t=60)
    )
    return graph


# Courbe
# Evolution chiffre d'affaire
def plot_evolution_chiffre_affaire(data) :
    df_plot = data.groupby(pd.Grouper(key='Transaction_Date', freq='W')).apply(calculer_chiffre_affaire)[:-1]
    chiffre_evolution = px.line(
        x=df_plot.index, y=df_plot,
        title="Evolution du chiffre d'affaire par semaine",
        labels={"x": "Semaine", "y": "Chiffre d'affaire"},
    ).update_layout( 
        width=1000, height=400,
        margin=dict(t=60, b=0),
        
    )
    return chiffre_evolution


# Indicateur 1
## Chiffre d'affaire du mois
def plot_chiffre_affaire_mois(data) :
    df_plot = indicateur_du_mois(data, freq=False)
    indicateur = go.Figure(
        go.Indicator(
            mode = "number+delta",
            value = df_plot[1],
            delta = {'reference': df_plot[0]},
            domain = {'row': 0, 'column': 1},
            title=f"{df_plot.index[1]}",
        )
    ).update_layout(
        width=200, height=200, 
        margin=dict(l=0, r=20, t=20, b=0)
    )
    return indicateur


# Indicateur 2
# Ventes du mois
def plot_vente_mois(data, abbr=False) :
    df_plot = indicateur_du_mois(data, freq=True, abbr=abbr)
    indicateur = go.Figure(
        go.Indicator(
            mode = "number+delta",
            value = df_plot[1],
            delta = {'reference': df_plot[0]},
            domain = {'row': 0, 'column': 1},
            title=f"{df_plot.index[1]}",
        )
    ).update_layout( 
        width=200, height=200, 
        margin=dict(l=0, r=20, t=20, b=0)
    )
    return indicateur


# Table
# Table des 100 dernières ventes
def table_100_derniere_ventes(data):
    df_plot_copy = data.copy()
    df_plot_copy['Transaction_Date'] = df_plot_copy['Transaction_Date'].dt.date
    df_plot_copy = df_plot_copy.sort_values('Transaction_Date', ascending=False).head(100)
    columns = [
        {"name": "Date", "id": "Transaction_Date"},
        {"name": "Gender", "id": "Gender"},
        {"name": "Location", "id": "Location"},
        {"name": "Product Category", "id": "Product_Category"},
        {"name": "Quantity", "id": "Quantity"},
        {"name": "Avg Price", "id": "Avg_Price"},
        {"name": "Discount Pct", "id": "Discount_pct"}
    ]
    return {
        'data': df_plot_copy.to_dict('records'),
        'columns': columns
    }

# ============ Layout =============== #
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('ECAP Store', style={'fontSize': '24px','fontweight': 'bold','color': '#333'}),
                style={'display': 'flex', 'alignItems': 'center','justifyContent': 'flex-start', 'height': '100%'},
                width=6),
        dbc.Col(
            html.Div([
                dcc.Dropdown(id='dropdown', 
                             options=[{'label': str(location).title(), 'value': location} for location in df['Location'].dropna().unique()],
                             multi=False,
                             searchable=True,
                             placeholder='Choississez des zones.',  # texte dans la barre interactive
                             disabled=False,
                             style = {'width': '80%','boderRadius': '4px'}
                             ),
                    ], style = {'width': '100%'}),
                    style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end', 'height': '100%'},
                    width=6),
            ], style={"height": "60px", "backgroundColor": "#add8e6"}),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([dcc.Graph(id='indicateur1', style={'width': '60%', 'height': '150px'}, responsive=True)], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}, width=6),
                dbc.Col([dcc.Graph(id='indicateur2', style={'width': '60%', 'height': '150px'}, responsive=True)], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}, width=6),
            ], style={"height": "30vh"}),
            dbc.Row([dcc.Graph(id='graph2', style={'width': '300%', 'height': '500px'}, responsive=True)], style={"height": "50vh"}),
        ], width=5),
        dbc.Col([
            dbc.Row([dcc.Graph(id='Courbe', style={'width': '200%', 'height': '100%'}, responsive=True)], style={"height": "50%"}),
            dbc.Row(children=[
                dbc.Col([
                html.H5(children='Table des 100 dernières ventes', style={'marginRight': '30px'}),
                dash_table.DataTable(id='Table', 
                                     style_table={'width': '100%', "height": "50%"},
                                     editable=False,
                                     filter_action='native',
                                     sort_action='native',
                                     page_action='native',
                                     page_current=0,
                                     page_size=10,
                                     **table_100_derniere_ventes(df),
                                     style_header={'backgroundColor': 'white','fontWeight': 'bold',
                                                   'border': '1px solid #ddd','textAlign': 'right'},
                                     style_cell={'textAlign': 'right','padding': '0px',
                                                 'border': '1px solid #ddd','backgroundColor': 'white',
                                                  'height': '10px','lineHeight': '10px',},
                                    ),
                ]),
            ], style={"height": "50%"}),  
        ], width=7),
    ], style={"height": "100vh"})
], fluid=True)

# ================= Callbacks ================ #

# Indicateur 1
@callback(Output('indicateur1', 'figure'),
          Input('dropdown', 'value'))
def update_indicateur1(selected_values):
    if not selected_values:
        df_plot = df
    else:
        df_plot = df[df['Location'] == selected_values]
    return plot_chiffre_affaire_mois(df_plot)


# Indicateur 2
@callback(Output('indicateur2', 'figure'),
          Input('dropdown', 'value'))
def update_indicateur2(selected_values):
    if not selected_values:
        df_plot = df
    else:
        df_plot = df[df['Location'] == selected_values]
    return plot_vente_mois(df_plot)


# Barplot
@callback(Output('graph2', 'figure'),
          Input('dropdown', 'value'))
def update_graph2(selected_values):
    if not selected_values:
        df_plot = df
    else:
        df_plot = df[df['Location'] == selected_values]
    return barplot_top_10_ventes(df_plot)


# Courbe
@callback(Output('Courbe', 'figure'),
          Input('dropdown', 'value'))
def update_courbe(selected_values):
    if not selected_values:
        df_plot = df
    else:
        df_plot = df[df['Location'] == selected_values]
    return plot_evolution_chiffre_affaire(df_plot)


# Table
@callback(Output('Table', 'data'),
          Input('dropdown', 'value'))
def update_table(selected_values):
    if not selected_values:
        df_plot = df
    else:
        df_plot = df[df['Location'] == selected_values]
    return table_100_derniere_ventes(df_plot)['data']


# ================ Run server ================== #
if __name__ == '__main__':
    app.run_server(debug=True, port=8054)


