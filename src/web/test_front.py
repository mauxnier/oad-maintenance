import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom CSS styles
colors = {
    "background": "#f7f7f7",
    "text": "#333333",
    "primary": "#ff6f00",  # Car-inspired primary color
    "secondary": "#1976d2",  # Car-inspired secondary color
}

app.layout = dbc.Container(
    fluid=True,
    style={"background-color": colors["background"]},
    children=[
        html.Div(
            className="header",
            children=[
                html.Img(
                    src="https://example.com/car-logo.png", className="logo"
                ),  # Replace with the car logo image URL
                html.H1(
                    "Maintenance de Véhicules",
                    className="text-center",
                    style={"color": colors["secondary"]},
                ),
            ],
        ),
        html.Div(
            className="content",
            children=[
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Organe :", style={"color": colors["text"]}),
                                dcc.Input(
                                    id="organe-input", type="text", className="mb-3"
                                ),
                                html.Label(
                                    "Contexte :", style={"color": colors["text"]}
                                ),
                                dcc.Input(
                                    id="contexte-input", type="text", className="mb-3"
                                ),
                                html.Label(
                                    "Symptôme :", style={"color": colors["text"]}
                                ),
                                dcc.Input(
                                    id="symptome-input", type="text", className="mb-3"
                                ),
                                html.Label(
                                    "ID de l'équipement :",
                                    style={"color": colors["text"]},
                                ),
                                dcc.Input(
                                    id="equipement-id-input",
                                    type="text",
                                    className="mb-3",
                                ),
                                html.Button(
                                    "Prédire",
                                    id="predict-button",
                                    className="mt-3",
                                    style={
                                        "background-color": colors["primary"],
                                        "color": "white",
                                    },
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            html.Div(id="prediction-output", className="mt-4"), md=6
                        ),
                    ],
                    className="mb-5",
                ),
            ],
        ),
    ],
)


@app.callback(
    dash.dependencies.Output("prediction-output", "children"),
    [
        dash.dependencies.Input("predict-button", "n_clicks"),
        dash.dependencies.State("organe-input", "value"),
        dash.dependencies.State("contexte-input", "value"),
        dash.dependencies.State("symptome-input", "value"),
        dash.dependencies.State("equipement-id-input", "value"),
    ],
)
def predict(n_clicks, organe, contexte, symptome, equipement_id):
    # Perform prediction using the input values and your developed model
    # Replace this section of code with your own prediction logic
    if n_clicks is None:
        return None

    prediction_text = f"""
        Prédiction :
        Système N1 : ...
        Système N2 : ...
        Système N3 : ...
        ODR Libellé : ...
        Type de travail : ...
    """

    return html.Div(
        className="prediction-output",
        children=[
            html.H3(
                "Résultat de la prédiction",
                className="text-center mb-4",
                style={"color": colors["secondary"]},
            ),
            html.Pre(
                prediction_text,
                className="prediction-text",
                style={"color": colors["text"]},
            ),
        ],
    )


if __name__ == "__main__":
    app.run_server(debug=True)
