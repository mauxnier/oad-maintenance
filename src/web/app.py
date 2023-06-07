import dash

# import dash_core_components as dcc
from dash import dcc

# import dash_html_components as html
from dash import html

# import dash_bootstrap_components as dbc
import dash_table
import plotly.express as px
import pyAgrum as gum


class MyApp:
    def __init__(self, model_name, var_features, var_targets, bn, data_df):
        self.app = dash.Dash(__name__)
        self.model_name = model_name
        self.var_features = var_features
        self.var_targets = var_targets
        self.bn = bn
        self.data_df = data_df

        self.app.layout = html.Div(
            [
                html.H1(model_name),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(var),
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
                    [dcc.Graph(id=f"{var}-graph") for var in var_targets],
                    style={"width": "65%", "float": "right", "display": "inline-block"},
                ),
                html.H3("Analyse de données"),
                dash_table.DataTable(
                    id="datatable",
                    columns=[
                        {"name": "Variable", "id": "Variable"},
                        {"name": "Probability", "id": "Probability"},
                        {"name": "System", "id": "System"},
                    ],
                    data=[],
                    style_table={"width": "30%"},
                    style_data={"whiteSpace": "normal", "height": "auto"},
                    style_cell={"textAlign": "left"},
                ),
            ]
        )

        @self.app.callback(
            [dash.dependencies.Output(f"{var}-graph", "figure") for var in var_targets]
            + [dash.dependencies.Output("datatable", "data")],
            [
                dash.dependencies.Input(f"{var}-dropdown", "value")
                for var in var_features
            ],
        )
        def update_graph(*var_features_values):
            bn_ie = gum.LazyPropagation(bn)

            ev = {var: value for var, value in zip(var_features, var_features_values)}
            bn_ie.setEvidence(ev)
            bn_ie.makeInference()

            prob_target = []
            datatable_data = []

            for idx, var in enumerate(var_targets):
                prob_target_var = bn_ie.posterior(var).topandas().droplevel(0)
                prob_fig = px.bar(prob_target_var)
                prob_target.append(prob_fig)

                # Get the top 5 maximum probability values and their variables
                top_5_values = prob_target_var.nlargest(5).reset_index()
                top_5_values.columns = ["Variable", "Probability"]
                top_5_values["Variable"] = top_5_values["Variable"].replace(
                    {"SYSTEM_N1": "N1", "SYSTEM_N2": "N2", "SYSTEM_N3": "N3"}
                )
                top_5_values[
                    "System"
                ] = f"System {idx + 1}"  # Add System column based on the graph index
                datatable_data += top_5_values.to_dict("records")

            return tuple(prob_target) + (datatable_data,)

    def run(self):
        self.app.run_server(debug=True)
