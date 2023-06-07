import pandas as pd
import pyAgrum as gum


def create_model(data_df, model_name, var_cat, var_features, var_targets, arcs):
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("data_df must be a valid DataFrame object.")

    if data_df is None:
        raise ValueError("data_df cannot be None.")

    for var in var_cat:
        if var not in data_df.columns:
            raise ValueError(f"Variable {var} is not present in the data_df.")

        data_df[var] = data_df[var].astype("category")

    # Créer les variables du BN
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

    return bn
