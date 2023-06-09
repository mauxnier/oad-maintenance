import pandas as pd
import modules.utils as utils
import pyAgrum as gum
import modules.pyagrum_extra as gum_extra
from sklearn.model_selection import train_test_split


class ModelTraining:
    def __init__(self, data_df, model_name, var_cat, var_features, var_targets, arcs):
        self.data_df = data_df
        self.model_name = model_name
        self.var_cat = var_cat
        self.var_features = var_features
        self.var_targets = var_targets
        self.arcs = arcs
        self.train_data = None
        self.test_data = None
        self.bn = None

    def preprocess_data(self, test_size=0.2, random_state=42):
        # Convertir les variables catégorielles en type 'category'
        for var in self.var_cat:
            self.data_df[var] = self.data_df[var].astype("category")

        # Récupérer chaque valeur unique de SIG_CONTEXTE en enlevant les "/"
        valeurs_uniques = set()
        for valeur in self.data_df["SIG_CONTEXTE"]:
            valeurs = valeur.split("/")
            for val in valeurs:
                valeurs_uniques.add(
                    val.strip()
                )  # Supprimer les espaces autour des valeurs
        print("Nombre de valeurs uniques de SIG_CONTEXTE: ", len(valeurs_uniques))

        # On separe les donnees en deux jeux de donnees : Un pour l'entrainement, l'autre pour le test
        train_data, test_data = train_test_split(
            self.data_df, test_size=test_size, random_state=random_state
        )

        print("Training set size:", len(train_data))
        print("Testing set size:", len(test_data))

        self.train_data = train_data
        self.test_data = test_data

        # Créer une variable discrète pour SIG_CONTEXTE
        # sig_contexte = gum.LabelizedVariable(
        #     "SIG_CONTEXTE", "contexte", valeurs_uniques
        # )

        # # L'encodage one-hot permet de créer des variables binaires distinctes pour chaque valeur unique
        # one_hot_factors = sig_contexte.getFactors()

        # # Ajouter les données à chaque facteur de l'encodage one-hot
        # for valeur in self.data_df["SIG_CONTEXTE"]:
        #     for factor in one_hot_factors:
        #         if factor.name() == valeur:
        #             factor.fillWith(1)
        #         else:
        #             factor.fillWith(0)

    def fit_model(self):
        self.preprocess_data()  # Prétraiter les données

        self.bn = utils.create_model(
            self.train_data,
            self.model_name,
            self.var_cat,
            self.var_features,
            self.var_targets,
            self.arcs,
        )

    def evaluate_model(self, show_progress=True):
        if self.test_data is None:
            raise ValueError("test_data cannot be None. Use preprocess_data() method.")

        if self.bn is None:
            raise ValueError("The model must be trained first. Use fit_model() method.")

        # Estimation de la performance
        for target in self.var_targets:
            pred = self.bn.predict(
                self.test_data[self.var_features],
                var_target=target,
                show_progress=show_progress,
            )

            accuracy = (self.test_data[target] == pred).mean()
            print("Accuracy for", target, ":", accuracy)
