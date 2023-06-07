from ia.model_training import ModelTraining
from ia.data_process import DataProcess
from web.front_app import WebApp

# Variable des catégories
var_cat = [
    "ODR_LIBELLE",
    "TYPE_TRAVAIL",
    "SYSTEM_N1",
    "SYSTEM_N2",
    "SYSTEM_N3",
    "SIG_ORGANE",
    "SIG_CONTEXTE",
    "SIG_OBS",
    "LIGNE",
    "MODELE",
    "MOTEUR",
    "CONSTRUCTEUR",
]

# Configuration du modèle 1
model_name = "modele 1 simple"
var_features = ["SIG_ORGANE"]  # Variables explicatives
var_targets = ["SYSTEM_N3"]  # Variables à expliquer
arcs = [("SIG_ORGANE", "SYSTEM_N3")]

# Configuration du modèle 2
# model_name = "modele 2 moyen"
# var_features = ["SIG_ORGANE", "SIG_OBS", "MODELE", "MOTEUR", "CONSTRUCTEUR"]
# var_targets = ["SYSTEM_N3", "SYSTEM_N1", "SYSTEM_N2", "ODR_LIBELLE", "TYPE_TRAVAIL"]
# arcs = [
#     ("SYSTEM_N3", "SIG_OBS"),
#     ("SYSTEM_N3", "SIG_ORGANE"),
#     ("SYSTEM_N3", "MODELE"),
#     ("SYSTEM_N3", "MOTEUR"),
#     ("SYSTEM_N3", "CONSTRUCTEUR"),
#     ("SYSTEM_N3", "TYPE_TRAVAIL"),
#     ("SYSTEM_N3", "ODR_LIBELLE"),
#     ("TYPE_TRAVAIL", "ODR_LIBELLE"),
#     ("SYSTEM_N3", "SYSTEM_N2"),
#     ("SYSTEM_N2", "SYSTEM_N1"),
# ]

# Charger les données
data = DataProcess()
data.run()

# Entrainer le modèle
modele = ModelTraining(
    data.data_df, model_name, var_cat, var_features, var_targets, arcs
)
modele.fit_model()  # Entrainer le modèle
# modele.evaluate_model()  # Evaluer le modèle

# Lancement de l'application
if __name__ == "__main__":
    # Créer une instance de la classe MyApp
    web_app = WebApp(
        model_name="OAD - Système de recommandation de système de réparation automobile",
        var_features=var_features,
        var_targets=var_targets,
        data_df=modele.train_data,
        bn=modele.bn,
    )

    # Lancer l'application
    web_app.run()
