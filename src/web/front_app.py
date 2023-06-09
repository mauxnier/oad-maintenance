import dash
from dash import dcc
from dash import html

import plotly.graph_objects as go
import pyAgrum as gum


class WebApp:
    def __init__(self, model_name, var_features, var_targets, bn, data_df):
        self.app = dash.Dash(__name__)
        self.model_name = model_name
        self.var_features = var_features
        self.var_targets = var_targets
        self.bn = bn
        self.data_df = data_df
        label_names = {
            "SIG_ORGANE": "Organe Signalé",
            "SIG_OBS": "Observation Signalée",
            "MOTEUR": "Moteur du véhicule"}
                       

        self.app.layout = html.Div(
            [
                html.H1(model_name),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(label_names[var]),
                                dcc.Dropdown(
                                    id=f"{var}-dropdown",
                                    options=[
                                        {"label": i, "value": i}
                                        for i in data_df[var].cat.categories
                                    ],
                                    value=data_df[var].cat.categories[0],
                                ),
                            ]
                        )
                        for var in var_features
                    ],
                    style={"width": "30%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Graph(id=f"{var}-graph", config={"displayModeBar": False})
                        for var in var_targets
                    ],
                    style={"width": "65%", "float": "right", "display": "inline-block"},
                ),
            ]
        )

        @self.app.callback(
            [dash.dependencies.Output(f"{var}-graph", "figure") for var in var_targets],
            [dash.dependencies.Input(f"{var}-dropdown", "value") for var in var_features],
        )
        def update_graph(*var_features_values):
            bn_ie = gum.LazyPropagation(bn)

            ev = {var: value for var, value in zip(var_features, var_features_values)}
            bn_ie.setEvidence(ev)
            bn_ie.makeInference()

            prob_target = []

            for idx, var in enumerate(var_targets):
                prob_target_var = bn_ie.posterior(var).topandas().droplevel(0)
                top_5_values = prob_target_var.nlargest(5)
                var_names = {"SYSTEM_N1": "Système de niveau 1",
                             "SYSTEM_N2": "Système de niveau 2",
                             "SYSTEM_N3": "Système de niveau 3",
                             "TYPE_TRAVAIL": "Type de travail"}

                # Create bar chart figure
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=top_5_values.index,
                        y=top_5_values.values,
                        marker_color="blue",
                    )
                )
                fig.update_layout(
                    title=f"Prédiction : {var_names[var]}",
                    xaxis_title="Valeur",
                    yaxis_title="Probabilité",
                )
                prob_target.append(fig)

            return tuple(prob_target)

    def run(self):
        self.app.run_server(debug=True)
