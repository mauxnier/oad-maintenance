import pandas as pd
import os
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
from pyagrum_extra import gum

dataFolder = "./data"

ot_odr_filename = os.path.join(dataFolder, "OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        compression="bz2",
                        sep=";")

equipements_filename = os.path.join(dataFolder, 'EQUIPEMENTS.csv')
equipements_df = pd.read_csv(equipements_filename,
                             sep=";")

print(equipements_df.head())
print(ot_odr_df.columns)

# On commence par faire un merge entre les deux tables
data_df = pd.merge(ot_odr_df, equipements_df, on="EQU_ID")
print(data_df.columns)


# On separe les donnees en deux jeux de donnees : Un pour l'entrainement, l'autre pour le test
from sklearn.model_selection import train_test_split
import pyAgrum as gum

train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)

print("Training set size:", len(train_data))
print("Testing set size:", len(test_data))


var_cat = ['ODR_LIBELLE', 'TYPE_TRAVAIL',
           'SYSTEM_N1', 'SYSTEM_N2', 'SYSTEM_N3', 
           'SIG_ORGANE', 'SIG_CONTEXTE', 'SIG_OBS', 'LIGNE', 'MODELE', 'MOTEUR', 'CONSTRUCTEUR']
for var in var_cat:
    data_df[var] = data_df[var].astype('category')


# Configuration du modèle
model_name = "modèle simple"
var_features = ["SIG_CONTEXTE", "SIG_ORGANE", "SIG_OBS", "MODELE", "MOTEUR", "CONSTRUCTEUR", ] # Variables explicatives
var_targets = ["SYSTEM_N3", "SYSTEM_N1", "SYSTEM_N2"] # Variables à expliquer
arcs = [("SYSTEM_N3", "SIG_OBS"),
        ("SYSTEM_N3", "SIG_CONTEXTE"),
        ("SYSTEM_N3", "SIG_ORGANE"),
        ("SYSTEM_N3", "MODELE"),
        ("SYSTEM_N3", "MOTEUR"),
        ("SYSTEM_N3", "CONSTRUCTEUR"),
        ("SYSTEM_N2", "SYSTEM_N3"),
        ("SYSTEM_N1", "SYSTEM_N2")]

# Création du modèle
var_to_model = var_features + var_targets
var_bn = {}
for var in var_to_model:
    nb_values = len(data_df[var].cat.categories)
    var_bn[var] = gum.LabelizedVariable(var, var, nb_values)

for var in var_bn:
    for i, modalite in enumerate(data_df[var].cat.categories):
        var_bn[var].changeLabel(i, modalite)

# On cree un BN avec pyAgrum
bn = gum.BayesNet(model_name)

for var in var_bn.values():
    bn.add(var)

for arc in arcs:
    bn.addArc(*arc)

# Apprentissage des LPC
bn.fit_bis(data_df, verbose_mode=True)






# Création de l'application
# =========================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(model_name),
    html.Div([
        html.Div([
            html.Label(var),
            dcc.Dropdown(
                id=f'{var}-dropdown',
                options=[{'label': i, 'value': i} for i in data_df[var].cat.categories],
                value=data_df[var].cat.categories[0]
            )
        ]) for var in var_features],
             style={'width': '30%', 'display': 'inline-block'}),
    html.Div([
            dcc.Graph(id=f'{var}-graph') 
        for var in var_targets],
             style={'width': '65%', 'float': 'right', 'display': 'inline-block'})
    ])


@app.callback(
    [Output(f'{var}-graph', 'figure') for var in var_targets],
    [Input(f'{var}-dropdown', 'value') for var in var_features]
)
def update_graph(*var_features_values):
    bn_ie = gum.LazyPropagation(bn)

    ev = {var: value for var, value in zip(var_features, var_features_values)}
    bn_ie.setEvidence(ev)
    bn_ie.makeInference()

    prob_target = []
    for var in var_targets:
        prob_target_var = bn_ie.posterior(var).topandas().droplevel(0)
        prob_fig = px.bar(prob_target_var)
        prob_target.append(prob_fig)
        
    return tuple(prob_target)


if __name__ == '__main__':
    app.run_server(debug=True)






