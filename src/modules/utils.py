import pandas as pd
import pyAgrum as gum


def create_model(train_df, model_name, var_features, var_targets, arcs):
    if not isinstance(train_df, pd.DataFrame):
        raise ValueError("train_df must be a valid DataFrame object.")

    if train_df is None:
        raise ValueError("train_df cannot be None.")

    # Créer les variables du BN
    var_to_model = var_features + var_targets
    var_bn = {}
    for var in var_to_model:
        nb_values = len(train_df[var].cat.categories)
        var_bn[var] = gum.LabelizedVariable(var, var, nb_values)

    for var in var_bn:
        for i, modalite in enumerate(train_df[var].cat.categories):
            var_bn[var].changeLabel(i, str(modalite))

    # On cree un BN avec pyAgrum
    bn = gum.BayesNet(model_name)

    for var in var_bn.values():
        bn.add(var)

    for arc in arcs:
        bn.addArc(*arc)

    # Apprentissage des LPC
    bn.fit_bis(train_df, verbose_mode=True)

    return bn


def generate_arcs(sig_contexte, target="SYSTEM_N3", target_to_sig=True):
    arcs = []
    for contexte in sig_contexte:
        if target_to_sig:
            arcs.append((target, contexte))
        else:
            arcs.append((contexte, target))
    return arcs
