from ia.model_training import ModelTraining
from ia.data_process import DataProcess
from web.front_app import WebApp
import modules.utils as utils

# Valeurs uniques du SIG_CONTEXTE
sig_contexte = [
    "LATERAL",
    "EN VIRAGE",
    "BAS",
    "A LA FERMETURE",
    "PORTE",
    "GAUCHE",
    "CENTRALE",
    "AU FREINAGE",
    "PLAFOND",
    "EN DESCENTE",
    "POSTE CONDUITE",
    "INTERIEUR",
    "A L'OUVERTURE",
    "AU POINT MORT",
    "CENTRE",
    "A FROID",
    "ARRIERE",
    "DESSOUS",
    "AU RALENTI",
    "PLATE FORME",
    "A L'ACCELERATION",
    "A CHAUD",
    "A VIDE",
    "EN MONTEE",
    "ROTONDE",
    "DROIT",
    "EXTERIEUR",
    "TABLEAU DE BORD",
    "AVANT",
    "EN CHARGE",
    "HAUT",
    "REMORQUE",
    "AU DEMARRAGE",
    "A L'ARRET",
]

# Variable des catégories
var_cat = [
    "ODR_LIBELLE",
    "TYPE_TRAVAIL",
    "SYSTEM_N1",
    "SYSTEM_N2",
    "SYSTEM_N3",
    "SIG_ORGANE",
    "SIG_OBS",
    "LIGNE",
    "MODELE",
    "MOTEUR",
    "CONSTRUCTEUR",
]

# Nom des variables explicatives dans l'application
features_names = {
    "SIG_ORGANE": "Organe Signalé",
    "SIG_OBS": "Observation Signalée",
    "SIG_CONTEXTE": "Contexte Signalé",
    "MOTEUR": "Moteur du véhicule",
}

# Nom des variables explicatives dans l'application
targets_names = {
    "SYSTEM_N1": "Système de niveau 1",
    "SYSTEM_N2": "Système de niveau 2",
    "SYSTEM_N3": "Système de niveau 3",
    "TYPE_TRAVAIL": "Type de travail",
    "ODR_LIBELLE": "Libellé de la réparation",
}

# Configuration des modèles simples
# model_name = "modele 1 simple"
# var_features = ["SIG_ORGANE"]  # Variables explicatives
# var_targets = ["SYSTEM_N3"]  # Variables à expliquer
# arcs = [("SIG_ORGANE", "SYSTEM_N3")]

# model_name = "modele 2 simple"
# var_features = ["SIG_ORGANE"]
# var_targets = ["SYSTEM_N3", "SYSTEM_N2", "SYSTEM_N1"]
# arcs = [("SYSTEM_N3", "SIG_ORGANE"),
#         ("SYSTEM_N3", "SYSTEM_N2"),
#         ("SYSTEM_N2", "SYSTEM_N1")
#         ]

# Configuration des modèles complexes
# model_name = "modele complexe"
# var_features = ["SIG_ORGANE", "SIG_OBS", "MOTEUR"]
# var_targets = ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3", "TYPE_TRAVAIL", "ODR_LIBELLE"]
# arcs = [
#     ("SYSTEM_N3", "SIG_ORGANE"),
#     ("SYSTEM_N3", "SIG_OBS"),
#     ("SYSTEM_N3", "MOTEUR"),
#     ("SYSTEM_N3", "SYSTEM_N2"),
#     ("SYSTEM_N2", "SYSTEM_N1"),
#     ("SYSTEM_N3", "TYPE_TRAVAIL"),
#     ("SYSTEM_N3", "ODR_LIBELLE"),
#     ("SIG_ORGANE", "ODR_LIBELLE"),
# ]

# model_name = "modele complexe sans odr_libelle"
# var_features = ["SIG_ORGANE", "SIG_OBS", "MOTEUR"]
# var_targets = ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3","TYPE_TRAVAIL"]
# arcs = [("SYSTEM_N3", "SIG_ORGANE"),
#         ("SYSTEM_N3", "SIG_OBS"),
#         ("SYSTEM_N3", "MOTEUR"),
#         ("SYSTEM_N3", "SYSTEM_N2"),
#         ("SYSTEM_N2", "SYSTEM_N1"),
#         ("SYSTEM_N3", "TYPE_TRAVAIL")]

# model_name = "modele sig_contexte"
# var_features = ["SIG_ORGANE"]  # Variables explicatives
# var_targets = ["SYSTEM_N3"]  # Variables à expliquer
# arcs = [("SIG_ORGANE", "SYSTEM_N3")]
# arcs = arcs + utils.generate_arcs(
#     sig_contexte, target="SYSTEM_N3", target_to_sig=True
# )  # ("SYSTEM_N3", sig_contexte)
# print(arcs)

# /!\ Ne pas oublier d'ajouter sig_contexte dans le ModelTraining et WebApp
model_name = "modele complexe avec sig_contexte"
var_features = ["SIG_ORGANE", "SIG_OBS", "MOTEUR"]
var_targets = ["SYSTEM_N1", "SYSTEM_N2", "SYSTEM_N3", "TYPE_TRAVAIL", "ODR_LIBELLE"]
arcs = [
    # Pour N3
    ("SYSTEM_N3", "SIG_ORGANE"),
    ("SYSTEM_N3", "SIG_OBS"),
    ("SYSTEM_N3", "MOTEUR"),
    ("SYSTEM_N3", "EN VIRAGE"),
    ("SYSTEM_N3", "A LA FERMETURE"),
    ("SYSTEM_N3", "AU FREINAGE"),
    ("SYSTEM_N3", "EN DESCENTE"),
    ("SYSTEM_N3", "A L'OUVERTURE"),
    ("SYSTEM_N3", "AU POINT MORT"),
    ("SYSTEM_N3", "A FROID"),
    ("SYSTEM_N3", "AU RALENTI"),
    ("SYSTEM_N3", "PLATE FORME"),
    ("SYSTEM_N3", "A L'ACCELERATION"),
    ("SYSTEM_N3", "A CHAUD"),
    ("SYSTEM_N3", "A VIDE"),
    ("SYSTEM_N3", "EN MONTEE"),
    ("SYSTEM_N3", "EN CHARGE"),
    ("SYSTEM_N3", "AU DEMARRAGE"),
    ("SYSTEM_N3", "A L'ARRET"),
    # Pour N2 et N1
    ("SYSTEM_N3", "SYSTEM_N2"),
    ("SYSTEM_N2", "SYSTEM_N1"),
    # Pour Type de travail et ODR
    ("SYSTEM_N3", "TYPE_TRAVAIL"),
    # ("SYSTEM_N3", "ODR_LIBELLE"),  # A vérifier changement de sens
    # ("SIG_ORGANE", "ODR_LIBELLE"),  # A vérifier changement de sens
    ("ODR_LIBELLE", "SYSTEM_N3"),
    ("ODR_LIBELLE", "SIG_ORGANE"),
    ("ODR_LIBELLE", "TYPE_TRAVAIL"),
    # Pour sig_contexte
    ("ODR_LIBELLE", "LATERAL"),
    ("ODR_LIBELLE", "BAS"),
    ("ODR_LIBELLE", "GAUCHE"),
    ("ODR_LIBELLE", "CENTRALE"),
    ("ODR_LIBELLE", "PLAFOND"),
    ("ODR_LIBELLE", "PORTE"),
    ("ODR_LIBELLE", "POSTE CONDUITE"),
    ("ODR_LIBELLE", "INTERIEUR"),
    ("ODR_LIBELLE", "CENTRE"),
    ("ODR_LIBELLE", "ARRIERE"),
    ("ODR_LIBELLE", "DESSOUS"),
    ("ODR_LIBELLE", "DROIT"),
    ("ODR_LIBELLE", "EXTERIEUR"),
    ("ODR_LIBELLE", "TABLEAU DE BORD"),
    ("ODR_LIBELLE", "AVANT"),
    ("ODR_LIBELLE", "HAUT"),
    ("ODR_LIBELLE", "REMORQUE"),
    ("ODR_LIBELLE", "ROTONDE"),
]
# arcs = arcs + utils.generate_arcs(sig_contexte, target="SYSTEM_N3", target_to_sig=True)

# Charger les données
data = DataProcess()
data.run()

# Entrainer le modèle
modele = ModelTraining(
    data.data_df,
    model_name,
    var_cat + sig_contexte,  # Ajout du sig_contexte
    var_features + sig_contexte,  # Ajout du sig_contexte
    var_targets,
    arcs,
)
modele.fit_model()  # Entrainer le modèle
# modele.evaluate_model_all_targets()  # Evaluer le modèle
# modele.evaluate_model_a_target(target="ODR_LIBELLE")

# Lancement de l'application
if __name__ == "__main__":
    # Créer une instance de la classe WebApp
    web_app = WebApp(
        model_name="OAD - Système de recommandation de système de réparation automobile",
        var_features=var_features + ["SIG_CONTEXTE_ENCODED"],  # Ajout du sig_contexte
        var_targets=var_targets,
        data_df=modele.train_data,
        bn=modele.bn,
        sig_contexte=sig_contexte,
        features_names=features_names,
        targets_names=targets_names,
    )

    # Lancer l'application
    web_app.run()
