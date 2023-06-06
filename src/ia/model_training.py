import pandas as pd
import os

dataFolder = "./data"

ot_odr_filename = os.path.join(dataFolder, "OT_ODR.csv.bz2")
ot_odr_df = pd.read_csv(ot_odr_filename,
                        compression="bz2",
                        sep=";")

equipements_filename = os.path.join(dataFolder, 'EQUIPEMENTS.csv')
equipements_df = pd.read_csv(equipements_filename,
                             sep=";")

print(equipements_df.head())
print(ot_odr_df.columns)

# On commence par faire un merge entre les deux tables
merged_df = pd.merge(ot_odr_df, equipements_df, on="EQU_ID")
print(merged_df.columns)


# On separe les donnees en deux jeux de donnees : Un pour l'entrainement, l'autre pour le test
from sklearn.model_selection import train_test_split
import pyAgrum as gum

train_data, test_data = train_test_split(merged_df, test_size=0.2, random_state=42)

print("Training set size:", len(train_data))
print("Testing set size:", len(test_data))

# On cree un BN avec pyAgrum
bn = gum.BayesNet("modèle simple")

# On cree les variables
bn.add(gum.LabelizedVariable("MODELE", "MODELE", merged_df["MODELE"].unique()))
bn.add(gum.LabelizedVariable("CONSTRUCTEUR", "CONSTRUCTEUR", merged_df["CONSTRUCTEUR"].unique()))
bn.add(gum.LabelizedVariable("MOTEUR", "MOTEUR", merged_df["MOTEUR"].unique()))
bn.add(gum.LabelizedVariable("SIG_ORGANE", "SIG_ORGANE", merged_df["SIG_ORGANE"].unique()))
bn.add(gum.LabelizedVariable("SIG_OBS", "SIG_OBS", merged_df["SIG_OBS"].unique()))
bn.add(gum.LabelizedVariable("SYSTEM_N1", "SYSTEM_N1", merged_df["SYSTEM_N1"].unique()))


# On cree les liens entre les variables
#bn.addArc("MODELE", "SYSTEM_N1")
#bn.addArc("CONSTRUCTEUR", "SYSTEM_N1")
#bn.addArc("MOTEUR", "SYSTEM_N1")
bn.addArc("SIG_ORGANE", "SYSTEM_N1")
#bn.addArc("SIG_OBS", "SYSTEM_N1")

ie = gum.LazyPropagation(bn)
ie.makeInference()

predictions = ie.posterior("SYSTEM_N1")


# Créer un classifieur basé sur le réseau bayésien
#classifier = gum.BNLearner(train_data, bn)

# Effectuer l'apprentissage du réseau bayésien à partir des données d'entraînement
#classifier.learnBN()

# Récupérer les variables cibles du réseau bayésien
target_variables = ['SYSTEM_N1']

# Récupérer les données de test pour les variables cibles
test_data_target = test_data[target_variables]

# Effectuer les prédictions sur les données de test
#predictions = classifier.predict(test_data)

# Évaluer les performances du modèle
accuracy = (predictions == test_data_target).mean()
print("Accuracy:", accuracy)







