import dash
from dash import dcc
from dash import html

import plotly.graph_objects as go
import pyAgrum as gum


class WebApp:
    def __init__(
        self,
        model_name,
        var_features,
        var_targets,
        data_df,
        bn,
        sig_contexte,
        features_names,
        targets_names,
    ):
        self.app = dash.Dash(__name__)
        self.model_name = model_name
        self.var_features = var_features
        self.var_targets = var_targets
        self.data_df = data_df
        self.bn = bn
        self.sig_contexte = sig_contexte
        self.features_names = features_names
        self.targets_names = targets_names

        html_components = []
        for var in self.var_features:
            if var == "SIG_CONTEXTE_ENCODED":
                html_components.extend(
                    [
                        html.Label(self.features_names["SIG_CONTEXTE"]),
                        dcc.Dropdown(
                            id=f"{var}-dropdown",
                            options=[
                                {"label": valeur, "value": valeur}
                                for valeur in self.sig_contexte
                            ],
                            value=None,
                            multi=True,  # Permet la sélection multiple
                        ),
                    ]
                )
            else:
                html_components.extend(
                    [
                        html.Label(self.features_names[var]),
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

        # print(html_components)

        self.app.layout = html.Div(
            style={
                "background-image": "url('src/web/background.jpg')",  # TODO Ajout image
                "background-size": "cover",
                "background-repeat": "no-repeat",
                "background-position": "center",
            },
            children=[
                html.H1(model_name),
                html.Div(
                    html_components, style={"width": "30%", "display": "inline-block"}
                ),
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(
                                id=f"{var}-graph", config={"displayModeBar": False}
                            ),
                            style={"width": "33%", "display": "inline-block"},
                        )
                        for var in self.var_targets
                    ],
                    style={"width": "100%", "display": "flex", "flexWrap": "wrap"},
                ),
            ],
        )

        @self.app.callback(
            [
                dash.dependencies.Output(f"{var}-graph", "figure")
                for var in self.var_targets
            ],
            [
                dash.dependencies.Input(f"{var}-dropdown", "value")
                for var in self.var_features
            ],
        )
        def update_graph(*var_features_values):
            print("var_features_values: ", var_features_values)
            ev = {}
            # Vérifier si "SIG_CONTEXTE_ENCODED" est présent dans les variables sélectionnées
            if "SIG_CONTEXTE_ENCODED" in var_features:
                # Récupérer l'index de "SIG_CONTEXTE_ENCODED" dans var_features
                index = self.var_features.index("SIG_CONTEXTE_ENCODED")

                # Récupérer les valeurs sélectionnées de la liste déroulante "SIG_CONTEXTE_ENCODED"
                selected_values = var_features_values[index]
                # print("selected_values: ", selected_values)

                if selected_values:
                    # Définir les valeurs correspondantes dans l'évidence
                    for valeur in self.sig_contexte:
                        if valeur in selected_values:
                            ev[valeur] = 1
                #         else:
                #             ev[valeur] = 0
                # else:
                #     # Si la liste est vide, définir toutes les valeurs à 0
                #     for valeur in self.sig_contexte:
                #         ev[valeur] = 0

                # Supprimer "SIG_CONTEXTE_ENCODED" de la liste var_features
                var_features_values_except = (
                    var_features_values[:index] + var_features_values[index + 1 :]
                )
                print(var_features_values_except)

                ev.update(
                    {
                        var: value
                        for var, value in zip(
                            var_features[:index] + var_features[index + 1 :],
                            var_features_values_except,
                        )
                    }
                )
            else:
                # Définir les valeurs correspondantes dans l'évidence
                ev = {
                    var: value for var, value in zip(var_features, var_features_values)
                }
            print("ev: ", ev)

            # Créer un objet LazyPropagation
            bn_ie = gum.LazyPropagation(bn)

            # Définir l'évidence et faire l'inférence
            bn_ie.setEvidence(ev)
            bn_ie.makeInference()

            prob_target = []

            for idx, var in enumerate(var_targets):
                prob_target_var = bn_ie.posterior(var).topandas().droplevel(0)
                top_5_values = prob_target_var.nlargest(5)

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
                    title=f"Prédiction : {self.targets_names[var]}",
                    xaxis_title="Valeur",
                    yaxis_title="Probabilité",
                )
                prob_target.append(fig)

            return tuple(prob_target)

    def run(self):
        self.app.run_server(debug=True)
