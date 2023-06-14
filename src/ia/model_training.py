import pandas as pd
import modules.utils as utils
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

    def preprocess_data(self):
        # print(50 * "-")
        # print(self.data_df["SIG_CONTEXTE"].head())

        # Récupérer chaque valeur unique de SIG_CONTEXTE en enlevant les "/"
        # valeurs_uniques = set()
        # for valeur in self.data_df["SIG_CONTEXTE"]:
        #     valeurs = valeur.split("/")
        #     for val in valeurs:
        #         valeurs_uniques.add(
        #             val.strip()
        #         )  # Supprimer les espaces autour des valeurs
        # valeurs_uniques = list(valeurs_uniques)
        # print(50 * "-")
        # print("Valeurs uniques de SIG_CONTEXTE: ", valeurs_uniques)

        # Définir l'option display.max_columns sur None
        # pd.set_option("display.max_columns", None)

        # Séparer chaque valeur unique en colonnes distinctes: onehot encoding
        encoded_df = self.data_df["SIG_CONTEXTE"].str.get_dummies(sep="/")
        # print(50 * "-")
        # print(encoded_df.head())

        merged_df = pd.concat(
            [self.data_df.drop("SIG_CONTEXTE", axis=1), encoded_df], axis=1
        )
        # print(50 * "-")
        # print(merged_df.head())
        self.data_df = merged_df

        # Convertir les variables catégorielles en type 'category'
        for var in self.var_cat:
            if var not in self.data_df.columns:
                raise ValueError(f"Variable {var} is not present in the data_df.")
            self.data_df[var] = self.data_df[var].astype("category")

    def split_data(self, test_size=0.2, random_state=42):
        # On separe les donnees en deux jeux de donnees : Un pour l'entrainement, l'autre pour le test
        train_data, test_data = train_test_split(
            self.data_df, test_size=test_size, random_state=random_state
        )

        print("Training set size:", len(train_data))
        print("Testing set size:", len(test_data))

        self.train_data = train_data
        self.test_data = test_data

    def fit_model(self):
        self.preprocess_data()  # Prétraiter les données
        self.split_data()  # Séparer les données en train et test

        self.bn = utils.create_model(
            self.train_data,
            self.model_name,
            self.var_features,
            self.var_targets,
            self.arcs,
        )

    def evaluate_model_all_targets(self, show_progress=True):
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

    def evaluate_model_a_target(self, show_progress=True, target="SYSTEM_N3"):
        if self.test_data is None:
            raise ValueError("test_data cannot be None. Use preprocess_data() method.")

        if self.bn is None:
            raise ValueError("The model must be trained first. Use fit_model() method.")

        # Estimation de la performance
        pred = self.bn.predict(
            self.test_data[self.var_features],
            var_target=target,
            show_progress=show_progress,
        )

        accuracy = (self.test_data[target] == pred).mean()
        print("Accuracy for", target, ":", accuracy)
